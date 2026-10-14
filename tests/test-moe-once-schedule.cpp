// ranma: the PCIe-once expert-major op (MOE_FFN_HAD op_params[3], ggml/src/ggml-cuda/mul-mat-had.cu) against a simulated
// device, on the CPU. No GPU, no ggml.
//
// 1. The stream schedule. The op issues, per op: on the compute stream the routing, a record of ev_plan that the side
//    stream waits for, the plan and one host synchronize; the copies of chunks 0 and 1; then per chunk c: a wait for
//    ev_ready[c], the GEMMs (they read the staged experts of chunk c in buffer c % 2, and the exact op writes the down
//    rows of the chunk to T), the release, a record of ev_cdone[c], the copies of chunk c + 2 (on the side stream after a
//    wait for ev_cdone[c]) and (partial) the scatter of the chunk; (exact) one scatter that reads every row of T; at the
//    end the side stream records ev_done and the compute stream waits for it. The two buffers and T come from the pool
//    and are returned at the end of the op: the next user of that memory is a later compute stream command.
//    The simulator runs this command list (transcribed from the op) with random durations and a random interleaving of
//    the two streams, CUDA event semantics (a wait waits for the last record issued before it; ev_ready and ev_cdone
//    are reused by the next op), several ops one after the other, and checks:
//      - a GEMM reads a buffer only when it holds all the copies of its chunk and no copy writes it meanwhile;
//      - a copy never overwrites a buffer a GEMM still has to read (the release is after the last consumer);
//      - T is written once per row before the scatter that reads it;
//      - every wait ends (no deadlock: some stream can always proceed until both are empty);
//      - the pool memory is reused only after the last copy into it finished.
//    Mutations of the schedule (a missing wait, an early release) must be caught.
// 2. The sums. The op sorts the rows (token, slot) by expert and cuts chunks of at most R rows and stage_cap staged
//    experts (the routing kernel's rule, transcribed). With T, every row keeps its sorted position and one scatter adds
//    the rows of a token in slot order: the result must be byte-equal to the one-chunk op of each ubatch. The partial
//    op (a sum per chunk, added in chunk order) is reported as information.
// What this does not show: the GPU itself (the copy engine, the events and the kernels), and that the GEMM gives the same
// row whatever its tile; the model-free op test (test-backend-ops pcie-once) checks those on the device.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

// ---- 1. the stream schedule ------------------------------------------------------------------------------------------

enum cmd_kind { K_RECORD, K_WAIT, K_COPY, K_GEMM, K_SCATTER_ALL, K_HOST_SYNC, K_POOL_REUSE };

struct cmd {
    cmd_kind k;
    int ev    = -1; // event id (record, wait)
    int chunk = -1;
    int buf   = -1;
    int n     = 0;  // staged experts of the chunk (copy, GEMM)
    int op    = 0;
    int seq   = -1; // wait: the record sequence it waits for (filled at issue)
};

struct mutation {
    bool no_side_wait   = false; // the copies of chunk c + 2 do not wait for ev_cdone[c]
    bool no_ready_wait  = false; // the GEMMs of chunk c do not wait for ev_ready[c]
    bool no_end_fence   = false; // the op does not wait for the side stream at its end
    bool early_release  = false; // ev_cdone[c] is recorded before the GEMMs of chunk c
};

struct sim {
    std::vector<cmd> q[2]; // 0 compute, 1 side
    std::vector<int> ev_seq;         // last record sequence issued per event
    std::vector<int> ev_done_seq;    // last record sequence completed per event
    int seq = 0;

    int new_event() {
        ev_seq.push_back(-1);
        ev_done_seq.push_back(-1);
        return (int) ev_seq.size() - 1;
    }
    void record(int s, int ev, int op) {
        cmd c{K_RECORD};
        c.ev = ev;
        c.seq = ++seq;
        c.op = op;
        ev_seq[ev] = c.seq;
        q[s].push_back(c);
    }
    void wait(int s, int ev, int op) {
        cmd c{K_WAIT};
        c.ev = ev;
        c.seq = ev_seq[ev]; // -1: never recorded, completes at once
        c.op = op;
        q[s].push_back(c);
    }
    void push(int s, cmd c) {
        q[s].push_back(c);
    }
};

// the commands of one op, in the order the host issues them (mul-mat-had.cu, ggml_cuda_moe_ffn_had with sd && sdma)
struct op_events {
    int plan, done, sync;
    std::vector<int> ready, cdone;
};

static void issue_op(sim & S, op_events & E, const std::vector<int> & stage_n, bool exact, int op, const mutation & m) {
    const int nc = (int) stage_n.size();
    S.record(0, E.plan, op);
    S.wait(1, E.plan, op);
    // the host waits until the compute stream is idle (the plan readback) before it issues the rest: the side stream
    // commands below cannot start before that point of the compute stream
    S.record(0, E.sync, op);
    S.wait(1, E.sync, op);
    auto enqueue_copies = [&](int c) {
        if (c >= nc) {
            return;
        }
        if (c >= 2 && !m.no_side_wait) {
            S.wait(1, E.cdone[c - 2], op);
        }
        cmd cp{K_COPY};
        cp.chunk = c;
        cp.buf = c & 1;
        cp.n = stage_n[c];
        cp.op = op;
        S.push(1, cp);
        S.record(1, E.ready[c], op);
    };
    enqueue_copies(0);
    enqueue_copies(1);
    for (int c = 0; c < nc; ++c) {
        if (!m.no_ready_wait) {
            S.wait(0, E.ready[c], op);
        }
        if (m.early_release) {
            S.record(0, E.cdone[c], op);
        }
        cmd g{K_GEMM};
        g.chunk = c;
        g.buf = c & 1;
        g.n = stage_n[c];
        g.op = op;
        S.push(0, g);
        if (!m.early_release) {
            S.record(0, E.cdone[c], op);
        }
        enqueue_copies(c + 2);
    }
    if (exact) {
        cmd sc{K_SCATTER_ALL};
        sc.op = op;
        S.push(0, sc);
    }
    S.record(1, E.done, op);
    if (!m.no_end_fence) {
        S.wait(0, E.done, op);
    }
    cmd r{K_POOL_REUSE}; // the next user of the pool memory (two buffers and T), on the compute stream
    r.op = op;
    S.push(0, r);
}

struct run_result {
    bool ok = true;
    std::string why;
};

// runs the queues with random durations; returns the first violation
static run_result run_sim(sim & S, int n_ops, const std::vector<std::vector<int>> & stage_ns, bool exact, std::mt19937_64 & rng) {
    run_result res;
    (void) exact;
    auto fail = [&](const std::string & w) {
        if (res.ok) {
            res.ok = false;
            res.why = w;
        }
    };
    struct running { bool on = false; int left = 0; cmd c; };
    running cur[2];
    size_t head[2] = { 0, 0 };
    // device state
    struct bufst { int op = -1, chunk = -1; bool writing = false; int readers = 0; };
    bufst buf[2];
    // the chunks of each op whose GEMMs ran (T rows written)
    std::vector<std::vector<char>> gemm_done(n_ops);
    for (int o = 0; o < n_ops; ++o) {
        gemm_done[o].assign(stage_ns[o].size(), 0);
    }
    std::vector<int> copies_in_flight_of_op(n_ops, 0);
    std::vector<int> copies_issued_of_op(n_ops, 0);
    std::vector<int> copies_total_of_op(n_ops, 0);
    for (int o = 0; o < n_ops; ++o) {
        copies_total_of_op[o] = (int) stage_ns[o].size();
    }

    auto can_start = [&](int s) -> bool {
        if (head[s] >= S.q[s].size()) {
            return false;
        }
        const cmd & c = S.q[s][head[s]];
        switch (c.k) {
            case K_WAIT:
                return c.seq < 0 || S.ev_done_seq[c.ev] >= c.seq;
            case K_HOST_SYNC:
                return true; // the compute stream is idle when it reaches this command (in order)
            default:
                return true;
        }
    };
    auto start = [&](int s) {
        cmd c = S.q[s][head[s]++];
        cur[s].on = true;
        cur[s].c = c;
        cur[s].left = (c.k == K_COPY || c.k == K_GEMM) ? 1 + (int) (rng() % 6) : 1;
        if (c.k == K_COPY) {
            bufst & b = buf[c.buf];
            if (c.n > 0) {
                if (b.readers > 0) {
                    fail("a copy writes a buffer a GEMM is reading (op " + std::to_string(c.op) + " chunk " + std::to_string(c.chunk) + ")");
                }
                // the chunk that used the buffer before (two before, or one of an earlier op) must have read it already
                if (b.op >= 0 && b.chunk >= 0 && !gemm_done[b.op][b.chunk] && stage_ns[b.op][b.chunk] > 0) {
                    fail("a copy overwrites staged experts not yet read (op " + std::to_string(c.op) + " chunk " + std::to_string(c.chunk) +
                         " over chunk " + std::to_string(b.chunk) + ")");
                }
                b.writing = true;
                b.op = c.op;
                b.chunk = c.chunk;
            }
            copies_in_flight_of_op[c.op]++;
            copies_issued_of_op[c.op]++;
        } else if (c.k == K_GEMM) {
            if (c.n > 0) {
                bufst & b = buf[c.buf];
                if (b.writing) {
                    fail("a GEMM reads a buffer while a copy writes it (op " + std::to_string(c.op) + " chunk " + std::to_string(c.chunk) + ")");
                }
                if (b.op != c.op || b.chunk != c.chunk) {
                    fail("a GEMM reads a buffer that does not hold its chunk (partial or stale slot; op " + std::to_string(c.op) +
                         " chunk " + std::to_string(c.chunk) + ")");
                }
                b.readers++;
            }
        } else if (c.k == K_SCATTER_ALL) {
            for (size_t j = 0; j < gemm_done[c.op].size(); ++j) {
                if (!gemm_done[c.op][j]) {
                    fail("the scatter of T runs before the GEMMs of a chunk");
                }
            }
        } else if (c.k == K_POOL_REUSE) {
            if (copies_in_flight_of_op[c.op] > 0) {
                fail("the pool memory of an op is reused while a copy into it runs");
            }
            if (copies_issued_of_op[c.op] < copies_total_of_op[c.op]) {
                fail("the pool memory of an op is reused before all its copies were issued");
            }
            for (bufst & b : buf) {
                if (b.writing) {
                    fail("the pool memory is reused while a buffer is being written");
                }
            }
        } else if (c.k == K_HOST_SYNC) {
            // the host waits for the compute stream: everything before it on the compute stream is done (in order)
        }
    };
    auto finish = [&](int s) {
        const cmd & c = cur[s].c;
        if (c.k == K_RECORD) {
            S.ev_done_seq[c.ev] = std::max(S.ev_done_seq[c.ev], c.seq);
        } else if (c.k == K_COPY) {
            if (c.n > 0) {
                buf[c.buf].writing = false;
            }
            copies_in_flight_of_op[c.op]--;
        } else if (c.k == K_GEMM) {
            if (c.n > 0) {
                buf[c.buf].readers--;
            }
            gemm_done[c.op][c.chunk] = 1;
        }
        cur[s].on = false;
    };

    for (int64_t tick = 0; ; ++tick) {
        if (tick > 1000000) {
            fail("no end (tick limit)");
            break;
        }
        bool progressed = false;
        for (int s = 0; s < 2; ++s) {
            if (!cur[s].on && can_start(s) && (rng() % 3) != 0) {
                start(s);
                progressed = true;
            }
        }
        for (int s = 0; s < 2; ++s) {
            if (cur[s].on && (rng() % 2) == 0) {
                if (--cur[s].left <= 0) {
                    finish(s);
                }
                progressed = true;
            }
        }
        const bool empty = head[0] >= S.q[0].size() && head[1] >= S.q[1].size() && !cur[0].on && !cur[1].on;
        if (empty) {
            break;
        }
        if (!progressed && !cur[0].on && !cur[1].on && !can_start(0) && !can_start(1)) {
            fail("deadlock: both streams wait");
            break;
        }
    }
    return res;
}

static bool schedule_tests(uint64_t seed0, int n_seeds, int & n_runs) {
    bool ok = true;
    // the plain schedule over random chunk counts and staged experts, several ops in a row, exact and partial
    for (int si = 0; si < n_seeds; ++si) {
        std::mt19937_64 rng(seed0 + (uint64_t) si);
        const int n_ops = 1 + (int) (rng() % 4);
        const bool exact = (rng() & 1) != 0;
        sim S;
        op_events E;
        E.plan = S.new_event();
        E.done = S.new_event();
        E.sync = S.new_event();
        std::vector<std::vector<int>> sns;
        int nc_max = 0;
        for (int o = 0; o < n_ops; ++o) {
            const int nc = 1 + (int) (rng() % 40);
            std::vector<int> sn(nc);
            for (int & x : sn) {
                x = (rng() % 4) == 0 ? 0 : 1 + (int) (rng() % 16);
            }
            sns.push_back(sn);
            nc_max = std::max(nc_max, nc);
        }
        for (int c = 0; c < nc_max; ++c) {
            E.ready.push_back(S.new_event()); // reused by every op, as the op's per-device events
            E.cdone.push_back(S.new_event());
        }
        for (int o = 0; o < n_ops; ++o) {
            issue_op(S, E, sns[o], exact, o, mutation{});
        }
        run_result r = run_sim(S, n_ops, sns, exact, rng);
        ++n_runs;
        if (!r.ok) {
            fprintf(stderr, "schedule seed %d: FAIL %s\n", si, r.why.c_str());
            ok = false;
        }
    }
    // mutations: each must be caught by some seed
    const char * names[4] = { "no side wait for ev_cdone", "no compute wait for ev_ready", "no end-of-op fence", "release before the GEMMs" };
    for (int mi = 0; mi < 4; ++mi) {
        mutation m;
        m.no_side_wait  = mi == 0;
        m.no_ready_wait = mi == 1;
        m.no_end_fence  = mi == 2;
        m.early_release = mi == 3;
        int caught = 0;
        for (int si = 0; si < 512 && caught == 0; ++si) {
            std::mt19937_64 rng(seed0 + 7777 + (uint64_t) si);
            sim S;
            op_events E;
            E.plan = S.new_event();
            E.done = S.new_event();
        E.sync = S.new_event();
            const int nc = 3 + (int) (rng() % 12);
            std::vector<int> sn(nc);
            for (int & x : sn) {
                x = 1 + (int) (rng() % 8);
            }
            for (int c = 0; c < nc; ++c) {
                E.ready.push_back(S.new_event());
                E.cdone.push_back(S.new_event());
            }
            issue_op(S, E, sn, true, 0, m);
            run_result r = run_sim(S, 1, { sn }, true, rng);
            caught += r.ok ? 0 : 1;
        }
        // the end-of-op fence is redundant while nothing aborts: the GEMMs of every chunk wait for its copies, so the
        // copies are done when the compute stream passes the last chunk (it stays for an op that stops early)
        const bool redundant = mi == 2;
        printf("mutation %-32s %s\n", names[mi], caught ? "caught" : redundant ? "not caught (redundant, expected)" : "NOT CAUGHT");
        ok = ok && (caught > 0 || redundant);
    }
    return ok;
}

// ---- 2. the sums -----------------------------------------------------------------------------------------------------

static float rowval(uint64_t g, int col) {
    uint64_t s = (g + 1)*0x9E3779B97F4A7C15ull ^ (uint64_t) (col + 1)*0xC2B2AE3D27D4EB4Full;
    s ^= s >> 29;
    s *= 0xBF58476D1CE4E5B9ull;
    s ^= s >> 32;
    return (float) ((double) (s >> 11)/(double) (1ull << 53)*8.0 - 4.0);
}

struct chunking {
    std::vector<int> rows_sorted, pos; // as the routing kernel: row g = k + n_used*t
    std::vector<int> cb, cn;           // chunk first sorted row, rows
};

// the routing kernel's sort (an arbitrary order inside an expert, as its atomics) and chunk rule (whole experts up to R
// rows and stage_cap staged experts; an expert of more rows takes chunks of its own)
static chunking route(const std::vector<int> & ids, int n_used, int n_exp, int R, int stage_cap, int stage_min, std::mt19937_64 & rng) {
    const int n_rows = (int) ids.size();
    chunking ch;
    std::vector<int> cnt(n_exp, 0), offs(n_exp, 0);
    for (int e : ids) {
        cnt[e]++;
    }
    for (int e = 1; e < n_exp; ++e) {
        offs[e] = offs[e - 1] + cnt[e - 1];
    }
    std::vector<std::vector<int>> per(n_exp);
    for (int g = 0; g < n_rows; ++g) {
        per[ids[g]].push_back(g);
    }
    ch.rows_sorted.assign(n_rows, -1);
    ch.pos.assign(n_rows, -1);
    for (int e = 0; e < n_exp; ++e) {
        std::shuffle(per[e].begin(), per[e].end(), rng);
        for (size_t i = 0; i < per[e].size(); ++i) {
            ch.rows_sorted[offs[e] + (int) i] = per[e][i];
            ch.pos[per[e][i]] = offs[e] + (int) i;
        }
    }
    int cur_b = 0, cur_n = 0, cur_s = 0;
    auto emit = [&](int b, int n) {
        if (n > 0) {
            ch.cb.push_back(b);
            ch.cn.push_back(n);
        }
    };
    for (int e = 0; e < n_exp; ++e) {
        const int c = cnt[e];
        if (c == 0) {
            continue;
        }
        if (c > R) {
            emit(cur_b, cur_n);
            cur_n = 0;
            cur_s = 0;
            for (int i = 0; i < c; i += R) {
                emit(offs[e] + i, std::min(R, c - i));
            }
            continue;
        }
        const bool st = stage_cap > 0 && c >= stage_min;
        if (cur_n + c > R || (st && cur_s == stage_cap)) {
            emit(cur_b, cur_n);
            cur_n = 0;
            cur_s = 0;
        }
        if (cur_n == 0) {
            cur_b = offs[e];
        }
        cur_s += st ? 1 : 0;
        cur_n += c;
    }
    emit(cur_b, cur_n);
    (void) n_used;
    return ch;
}

// out[t] += sum over the slots k of t whose sorted position is in [b, b + nr), in slot order (mmh_moe_scatter)
static void scatter(const chunking & ch, int b, int nr, const float * y, const std::vector<float> & w, int n_used, int n_tokens,
        int d, std::vector<float> & out) {
    for (int t = 0; t < n_tokens; ++t) {
        bool any = false;
        int p[64];
        for (int k = 0; k < n_used; ++k) {
            const int q = ch.pos[k + n_used*t] - b;
            p[k] = q >= 0 && q < nr ? q : -1;
            any = any || p[k] >= 0;
        }
        if (!any) {
            continue;
        }
        for (int col = 0; col < d; ++col) {
            float acc = 0.0f;
            for (int k = 0; k < n_used; ++k) {
                if (p[k] >= 0) {
                    acc += w[(size_t) k + (size_t) n_used*t]*y[(size_t) p[k]*d + col];
                }
            }
            out[(size_t) t*d + col] += acc;
        }
    }
}

static bool sum_tests(uint64_t seed0, int n_cases, int & n_bitequal, int & n_partial_diff) {
    bool ok = true;
    for (int ci = 0; ci < n_cases; ++ci) {
        std::mt19937_64 rng(seed0*31 + (uint64_t) ci);
        const int n_exp  = 8 + (int) (rng() % 120);
        const int n_used = 1 + (int) (rng() % std::min(8, n_exp));
        const int C      = 16 + (int) (rng() % 200);
        const int g      = 2 + (int) (rng() % 4);
        const int N      = C*(g - 1) + 1 + (int) (rng() % C); // the last ubatch may be short
        const int d      = 4 + (int) (rng() % 12);
        const int Rfull  = C*n_used;                              // the pass rows of the ubatch-major op (one chunk)
        const int R      = std::max(1, (int) (Rfull - (int) (rng() % (Rfull/2 + 1))));  // the PCIe-once op may lower R
        const int cap    = 1 + (int) (rng() % 16);
        const int smin   = 1 + (int) (rng() % 8);
        // routing: n_used distinct experts per token, skewed half the time
        std::vector<int> ids((size_t) N*n_used);
        const bool skew = (rng() & 1) != 0;
        for (int t = 0; t < N; ++t) {
            std::vector<int> pool(n_exp);
            for (int e = 0; e < n_exp; ++e) {
                pool[e] = e;
            }
            for (int k = 0; k < n_used; ++k) {
                int j;
                if (skew) {
                    const double u = (double) (rng() >> 11)/(double) (1ull << 53);
                    j = k + (int) (u*u*(n_exp - k));
                } else {
                    j = k + (int) (rng() % (uint64_t) (n_exp - k));
                }
                std::swap(pool[k], pool[std::min(j, n_exp - 1)]);
                ids[(size_t) t*n_used + k] = pool[k];
            }
        }
        std::vector<float> w((size_t) N*n_used);
        for (float & x : w) {
            x = (float) ((double) (rng() >> 11)/(double) (1ull << 53));
        }
        // y of a row depends only on the row (the GEMM result of row g = k + n_used*t)
        auto ybuf = [&](const chunking & ch, int b, int nr) {
            std::vector<float> y((size_t) nr*d);
            for (int r = 0; r < nr; ++r) {
                const int gr = ch.rows_sorted[b + r];
                for (int col = 0; col < d; ++col) {
                    y[(size_t) r*d + col] = rowval((uint64_t) gr, col);
                }
            }
            return y;
        };
        // reference: one op per ubatch, one chunk (R >= its rows), no staging
        std::vector<float> ref((size_t) N*d, 0.0f);
        for (int t0 = 0; t0 < N; t0 += C) {
            const int w_n = std::min(C, N - t0);
            std::vector<int> ids_u(ids.begin() + (size_t) t0*n_used, ids.begin() + (size_t) (t0 + w_n)*n_used);
            std::vector<float> w_u(w.begin() + (size_t) t0*n_used, w.begin() + (size_t) (t0 + w_n)*n_used);
            chunking ch = route(ids_u, n_used, n_exp, Rfull, 0, 1, rng);
            if (ch.cb.size() != 1) {
                fprintf(stderr, "sums case %d: the ubatch op is not one chunk\n", ci);
                ok = false;
            }
            // rows of the ubatch op are numbered inside it; y follows the global row
            std::vector<float> y((size_t) w_n*n_used*d);
            for (int r = 0; r < w_n*n_used; ++r) {
                const int gl = ch.rows_sorted[r];
                for (int col = 0; col < d; ++col) {
                    y[(size_t) r*d + col] = rowval((uint64_t) (gl + t0*n_used), col);
                }
            }
            std::vector<float> out((size_t) w_n*d, 0.0f);
            scatter(ch, 0, w_n*n_used, y.data(), w_u, n_used, w_n, d, out);
            std::copy(out.begin(), out.end(), ref.begin() + (size_t) t0*d);
        }
        // the PCIe-once op over all N tokens: chunks of R rows and cap staged experts
        chunking ch = route(ids, n_used, n_exp, R, cap, smin, rng);
        const int n_rows = N*n_used;
        // exact: T at the sorted positions, every row written once, one scatter over all rows
        std::vector<float> T((size_t) n_rows*d, std::nanf(""));
        std::vector<int> written(n_rows, 0);
        for (size_t c = 0; c < ch.cb.size(); ++c) {
            std::vector<float> y = ybuf(ch, ch.cb[c], ch.cn[c]);
            for (int r = 0; r < ch.cn[c]; ++r) {
                written[ch.cb[c] + r]++;
                memcpy(&T[(size_t) (ch.cb[c] + r)*d], &y[(size_t) r*d], (size_t) d*sizeof(float));
            }
        }
        for (int r = 0; r < n_rows; ++r) {
            if (written[r] != 1) {
                fprintf(stderr, "sums case %d: T row %d written %d times\n", ci, r, written[r]);
                ok = false;
                break;
            }
        }
        std::vector<float> ex((size_t) N*d, 0.0f);
        scatter(ch, 0, n_rows, T.data(), w, n_used, N, d, ex);
        const bool same = memcmp(ex.data(), ref.data(), ex.size()*sizeof(float)) == 0;
        if (!same) {
            fprintf(stderr, "sums case %d: exact not byte-equal (N %d C %d k %d R %d cap %d)\n", ci, N, C, n_used, R, cap);
            ok = false;
        } else {
            n_bitequal++;
        }
        // partial: a sum per chunk, in chunk order
        std::vector<float> pa((size_t) N*d, 0.0f);
        for (size_t c = 0; c < ch.cb.size(); ++c) {
            std::vector<float> y = ybuf(ch, ch.cb[c], ch.cn[c]);
            scatter(ch, ch.cb[c], ch.cn[c], y.data(), w, n_used, N, d, pa);
        }
        if (memcmp(pa.data(), ref.data(), pa.size()*sizeof(float)) != 0) {
            n_partial_diff++;
        }
    }
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    const uint64_t seed  = argc > 1 ? strtoull(argv[1], nullptr, 10) : 0;
    const int      n     = argc > 2 ? std::max(1, atoi(argv[2])) : 2000;
    int n_runs = 0;
    const bool ok1 = schedule_tests(seed, n, n_runs);
    int n_eq = 0, n_pd = 0;
    const int n_cases = std::max(50, n/10);
    const bool ok2 = sum_tests(seed, n_cases, n_eq, n_pd);
    printf("schedule: %d runs %s; sums: exact byte-equal %d of %d cases, partial differs in %d (information)\n", n_runs,
        ok1 ? "ok" : "FAIL", n_eq, n_cases, n_pd);
    printf("%s\n", ok1 && ok2 ? "PASS" : "FAIL");
    return ok1 && ok2 ? 0 : 1;
}
