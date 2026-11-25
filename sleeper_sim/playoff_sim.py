#!/usr/bin/env python3
"""Monte Carlo playoff simulation for a Sleeper fantasy league.

Each team's weekly score ~ Normal(optimal projected lineup total, sd), where sd is
the team's std dev of actual weekly scores (weeks 1..LAST_PLAYED) averaged 50/50
with the league-wide pooled within-team std dev.
"""
import json
import os
import sys
import time
import urllib.request

import numpy as np

LEAGUE_ID = "1339388729239834624"
SEASON = 2026
MY_ROSTER = 4
LAST_PLAYED = 11
REG_WEEKS = [12, 13, 14]
PO_WEEKS = [15, 16, 17]
N_SIMS = 100_000
SEED = 2026
POSITIONS = ["QB", "RB", "WR", "TE", "K", "DEF"]
SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2, "K": 1, "DEF": 1}
FLEX_OK = {"RB", "WR", "TE"}
INACTIVE_ALL_WEEKS = {"IR", "PUP", "Sus", "NA"}
INACTIVE_NEXT_WEEK = {"Out"}  # weekly designation: only trusted for the next game week

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
API = f"https://api.sleeper.app/v1/league/{LEAGUE_ID}"


def get(url, name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    req = urllib.request.Request(url, headers={"User-Agent": "playoff-sim/1.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            print(f"  retry {url}: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)
    with open(path, "w") as f:
        json.dump(data, f)
    time.sleep(0.5)
    return data


def projections(week):
    out = {}
    for pos in POSITIONS:
        url = (f"https://api.sleeper.com/projections/nfl/{SEASON}/{week}"
               f"?season_type=regular&position[]={pos}")
        for row in get(url, f"proj_w{week}_{pos}") or []:
            out[row["player_id"]] = row
    return out


def score(stats, scoring):
    """Recompute fantasy points from projected stats with the league's settings."""
    return sum(stats.get(k, 0) * v for k, v in scoring.items() if isinstance(stats.get(k), (int, float)))


def player_points(row, scoring):
    stats = row.get("stats") or {}
    pos = (row.get("player") or {}).get("position")
    # DEF/K projections give bucket stats inconsistently; Sleeper's pts_ppr already
    # matches this league for them, so only recompute for skill positions.
    if pos in ("QB", "RB", "WR", "TE") and stats:
        return score(stats, scoring)
    return stats.get("pts_ppr", 0.0) or 0.0


def optimal_lineup(roster_players, proj, scoring, week, reserve):
    """Highest-projected legal lineup. Greedy is optimal here: fill each dedicated
    slot with the best eligible players, then FLEX with the best remaining RB/WR/TE."""
    pool = []
    for pid in roster_players:
        row = proj.get(pid)
        if row is None:
            continue
        pl = row.get("player") or {}
        inj = pl.get("injury_status")
        if pid in reserve or inj in INACTIVE_ALL_WEEKS:
            continue
        if week == REG_WEEKS[0] and inj in INACTIVE_NEXT_WEEK:
            continue
        pos = pl.get("position")
        if pos not in POSITIONS:
            continue
        name = (f"{pl.get('first_name', '')} {pl.get('last_name', '')}".strip()
                if pos != "DEF" else f"{pid} DEF")
        pool.append({"pid": pid, "pos": pos, "name": name, "pts": player_points(row, scoring),
                     "opp": row.get("opponent"), "inj": inj})
    pool.sort(key=lambda p: -p["pts"])
    lineup, used = [], set()
    for slot in ["QB", "RB", "WR", "TE", "K", "DEF"]:
        picks = [p for p in pool if p["pos"] == slot and p["pid"] not in used][: SLOTS[slot]]
        for p in picks:
            used.add(p["pid"])
            lineup.append((slot, p))
    flex = [p for p in pool if p["pos"] in FLEX_OK and p["pid"] not in used][: SLOTS["FLEX"]]
    for p in flex:
        used.add(p["pid"])
        lineup.append(("FLEX", p))
    return sum(p["pts"] for _, p in lineup), lineup


def main():
    league = get(API, "league")
    rosters = get(API + "/rosters", "rosters")
    users = get(API + "/users", "users")
    scoring = league["scoring_settings"]

    uname = {u["user_id"]: (u.get("metadata") or {}).get("team_name") or u["display_name"] for u in users}
    teams = {}
    for r in rosters:
        s = r["settings"]
        teams[r["roster_id"]] = {
            "name": uname.get(r["owner_id"], f"Team {r['roster_id']}"),
            "players": r["players"] or [],
            "reserve": set(r.get("reserve") or []),
            "wins": s["wins"], "losses": s["losses"], "ties": s.get("ties", 0),
            "pf": s["fpts"] + s.get("fpts_decimal", 0) / 100,
        }
    rids = sorted(teams)
    idx = {rid: i for i, rid in enumerate(rids)}
    T = len(rids)

    # Variance from actual weekly scores
    hist = {rid: [] for rid in rids}
    for w in range(1, LAST_PLAYED + 1):
        for m in get(f"{API}/matchups/{w}", f"matchups_w{w}"):
            hist[m["roster_id"]].append(m["points"])
    team_sd = {rid: float(np.std(hist[rid], ddof=1)) for rid in rids}
    league_sd = float(np.sqrt(np.mean([team_sd[r] ** 2 for r in rids])))  # pooled within-team sd
    sd = np.array([0.5 * team_sd[r] + 0.5 * league_sd for r in rids])

    # Schedule for remaining regular-season weeks
    schedule = {}
    for w in REG_WEEKS:
        groups = {}
        for m in get(f"{API}/matchups/{w}", f"matchups_w{w}"):
            groups.setdefault(m["matchup_id"], []).append(m["roster_id"])
        schedule[w] = [tuple(g) for g in groups.values() if len(g) == 2]

    # Projected optimal scores
    proj_mean = {}
    lineups = {}
    for w in REG_WEEKS + PO_WEEKS:
        proj = projections(w)
        if len(proj) < 100:  # endpoint not populated for this week
            print(f"Week {w}: no projections available, using wk12-14 average", file=sys.stderr)
            continue
        for rid in rids:
            tot, lu = optimal_lineup(teams[rid]["players"], proj, scoring, w, teams[rid]["reserve"])
            proj_mean[(rid, w)] = tot
            lineups[(rid, w)] = lu
    for rid in rids:
        avg = np.mean([proj_mean[(rid, w)] for w in REG_WEEKS])
        teams[rid]["avg_proj"] = avg
        for w in PO_WEEKS:
            proj_mean.setdefault((rid, w), avg)

    # ---- Simulation ----
    rng = np.random.default_rng(SEED)
    weeks = REG_WEEKS + PO_WEEKS
    mu = np.array([[proj_mean[(rid, w)] for rid in rids] for w in weeks])  # (W, T)
    scores = rng.normal(mu[None, :, :], sd[None, None, :], size=(N_SIMS, len(weeks), T))

    wins = np.tile(np.array([teams[r]["wins"] + 0.5 * teams[r]["ties"] for r in rids], float), (N_SIMS, 1))
    pf = np.tile(np.array([teams[r]["pf"] for r in rids]), (N_SIMS, 1))
    my_w = np.zeros(N_SIMS, int)
    for wi, w in enumerate(REG_WEEKS):
        s = scores[:, wi, :]
        pf += s
        for a, b in schedule[w]:
            ia, ib = idx[a], idx[b]
            aw = s[:, ia] > s[:, ib]
            wins[:, ia] += aw
            wins[:, ib] += ~aw
            if MY_ROSTER in (a, b):
                my_w += aw if a == MY_ROSTER else ~aw

    # Seed: wins, then PF (PF breaks ties within the same win total)
    key = wins * 1e5 + pf
    order = np.argsort(-key, axis=1)  # order[:, k] = team index of seed k+1
    seeds = order[:, :6]
    rows = np.arange(N_SIMS)

    def play(a, b, wk):
        sa = scores[rows, wk, a]
        sb = scores[rows, wk, b]
        return np.where(sa > sb, a, b)

    q1 = play(seeds[:, 3], seeds[:, 4], 3)  # 4 v 5
    q2 = play(seeds[:, 2], seeds[:, 5], 3)  # 3 v 6
    f1 = play(seeds[:, 0], q1, 4)           # 1 v W(4v5)
    f2 = play(seeds[:, 1], q2, 4)           # 2 v W(3v6)
    champ = play(f1, f2, 5)

    def pct(mask_counts):
        return 100.0 * mask_counts / N_SIMS

    res = {}
    for rid in rids:
        i = idx[rid]
        res[rid] = {
            "playoff": pct((seeds == i).any(axis=1).sum()),
            "bye": pct((seeds[:, :2] == i).any(axis=1).sum()),
            "one": pct((seeds[:, 0] == i).sum()),
            "final": pct(((f1 == i) | (f2 == i)).sum()),
            "champ": pct((champ == i).sum()),
        }

    # ---- Output ----
    print(f"\n{league['name']} — {N_SIMS:,} simulations (weeks {REG_WEEKS[0]}-{PO_WEEKS[-1]})")
    print(f"League pooled weekly SD: {league_sd:.1f}\n")
    hdr = f"{'Team':<24}{'Rec':>7}{'PF':>9}{'AvgProj':>9}{'SD':>6}{'Playoff%':>10}{'Bye%':>7}{'#1%':>7}{'Final%':>8}{'Champ%':>8}"
    print(hdr)
    print("-" * len(hdr))
    for rid in sorted(rids, key=lambda r: (-res[r]["playoff"], -res[r]["champ"])):
        t, r = teams[rid], res[rid]
        rec = f"{t['wins']}-{t['losses']}" + (f"-{t['ties']}" if t["ties"] else "")
        star = "*" if rid == MY_ROSTER else " "
        print(f"{(star + t['name'])[:23]:<24}{rec:>7}{t['pf']:>9.1f}{t['avg_proj']:>9.1f}{sd[idx[rid]]:>6.1f}"
              f"{r['playoff']:>10.1f}{r['bye']:>7.1f}{r['one']:>7.1f}{r['final']:>8.1f}{r['champ']:>8.1f}")

    me = teams[MY_ROSTER]
    mi = idx[MY_ROSTER]
    print(f"\n{me['name']} — remaining schedule")
    from math import erf, sqrt
    for wi, w in enumerate(REG_WEEKS):
        a, b = next(m for m in schedule[w] if MY_ROSTER in m)
        opp = b if a == MY_ROSTER else a
        oi = idx[opp]
        m_mu, o_mu = proj_mean[(MY_ROSTER, w)], proj_mean[(opp, w)]
        z = (m_mu - o_mu) / sqrt(sd[mi] ** 2 + sd[oi] ** 2)
        p = 0.5 * (1 + erf(z / sqrt(2)))
        print(f"  Wk {w}: {m_mu:6.1f} vs {o_mu:6.1f}  {teams[opp]['name']:<24} win {100 * p:5.1f}%")

    in_po = (seeds == mi).any(axis=1)
    print(f"\n{me['name']} — playoff odds by record over weeks {REG_WEEKS[0]}-{REG_WEEKS[-1]}")
    for k in [3, 2, 1, 0]:
        m = my_w == k
        n = m.sum()
        final = f"{me['wins'] + k}-{me['losses'] + 3 - k}"
        po = 100 * in_po[m].mean() if n else float("nan")
        bye = 100 * (seeds[m, :2] == mi).any(axis=1).mean() if n else float("nan")
        print(f"  {k}-{3 - k} (→ {final}): playoff {po:5.1f}%  bye {bye:5.1f}%   [{100 * n / N_SIMS:4.1f}% of sims]")

    w = REG_WEEKS[0]
    print(f"\n{me['name']} — optimal week {w} lineup ({proj_mean[(MY_ROSTER, w)]:.1f} proj)")
    for slot, p in lineups[(MY_ROSTER, w)]:
        inj = f" ({p['inj']})" if p["inj"] else ""
        print(f"  {slot:<5}{p['name'][:24]:<25}{p['pos']:<4} vs {p['opp'] or 'BYE':<4}{p['pts']:6.1f}{inj}")
    starters = {p["pid"] for _, p in lineups[(MY_ROSTER, w)]}
    proj = projections(w)
    bench = []
    for pid in me["players"]:
        if pid in starters:
            continue
        row = proj.get(pid)
        if row is None:
            bench.append(f"{pid} (no proj)")
            continue
        pl = row["player"]
        nm = f"{pid} DEF" if pl.get("position") == "DEF" else f"{pl.get('first_name', '')} {pl.get('last_name', '')}"
        tag = f", {pl['injury_status']}" if pl.get("injury_status") else ""
        bench.append(f"{nm} {player_points(row, scoring):.1f}{tag}")
    print("  Bench: " + "; ".join(bench))


if __name__ == "__main__":
    main()
