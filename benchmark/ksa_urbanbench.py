"""KSA-UrbanBench: a Saudi urban-geography benchmark generated from national-address district data.

Ground truth is computed geometrically from official district boundaries, never written by hand.
Source data: github.com/zalabbad/Saudi-Arabia-Regions-Cities-and-Districts (collected from
maps.address.gov.sa), pinned to DATA_COMMIT for reproducibility.

Usage:  python ksa_urbanbench.py <path-to-data-repo> <output.jsonl> [items_per_task_per_city]
"""
import json, math, random, sys, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from collections import Counter
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union, transform
from shapely.validation import make_valid

DATA_COMMIT = "f9813f37ec006f52e99b9082a81bb7fa66f88dec"
VERSION = "1.0"
SEED = 2026
CITIES = {  # city_id -> (English, Arabic)
    "3": ("Riyadh", "الرياض"), "18": ("Jeddah", "جدة"), "6": ("Makkah", "مكة المكرمة"),
    "14": ("Madinah", "المدينة المنورة"), "13": ("Dammam", "الدمام"),
}
DIRS_EN = ["North", "Northeast", "East", "Southeast", "South", "Southwest", "West", "Northwest"]
DIRS_AR = ["شمال", "شمال شرق", "شرق", "جنوب شرق", "جنوب", "جنوب غرب", "غرب", "شمال غرب"]
TASKS = {  # code -> (dimension, answer type)
    "PIDK": ("KSA-Knowledge", "option"),     # point -> district, names only
    "DIRK": ("KSA-Knowledge", "option"),     # direction between two districts, names only
    "NEARK": ("KSA-Knowledge", "option"),    # nearest district, names only
    "ADJK": ("KSA-Knowledge", "yesno"),      # adjacency of two districts, names only
    "CITYK": ("KSA-Knowledge", "option"),    # district -> city
    "PIDC": ("KSA-Computation", "option"),   # point -> district, boundaries provided
}


# ----------------------------------------------------------------------------- geometry
def to_poly(rings):
    parts = []
    for r in rings:
        if len(r) >= 4:
            p = Polygon([(lon, lat) for lat, lon in r])      # source is (lat, lon)
            parts.append(make_valid(p) if not p.is_valid else p)
    g = unary_union(parts)
    return g if not g.is_empty else None


class Proj:
    """Local equirectangular projection (metres) around a city's reference latitude."""
    def __init__(self, lat0):
        self.kx = 111320.0 * math.cos(math.radians(lat0)); self.ky = 110540.0
    def fwd(self, g):
        return transform(lambda x, y, z=None: (x * self.kx, y * self.ky), g)
    def inv(self, g):
        return transform(lambda x, y, z=None: (x / self.kx, y / self.ky), g)


def bearing(lon1, lat1, lon2, lat2):
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def simplify_for_prompt(gm, proj, max_vertices=14):
    """Simplify a metric polygon until its exterior has <= max_vertices; returns lon/lat polygon."""
    g = gm if gm.geom_type == "Polygon" else max(gm.geoms, key=lambda q: q.area)
    tol = 20.0
    while True:
        s = g.simplify(tol, preserve_topology=True)
        if len(s.exterior.coords) - 1 <= max_vertices or tol > 2000:
            return s, proj.inv(s)
        tol *= 1.5


def fmt_poly(p_ll):
    return "[" + ", ".join(f"({y:.5f}, {x:.5f})" for x, y in list(p_ll.exterior.coords)[:-1]) + "]"


# ----------------------------------------------------------------------------- prompts
INSTR = {
    ("option", "en"): "\n\nYou may reason before answering. The last line of your reply must be the number of the correct option, e.g. 'Answer: 3'.",
    ("option", "ar"): "\n\nيمكنك التفكير قبل الإجابة. يجب أن يكون السطر الأخير من ردك رقم الخيار الصحيح بالصيغة 'Answer: 3'.",
    ("yesno", "en"): "\n\nYou may reason before answering. The last line of your reply must be 'Answer: Yes' or 'Answer: No'.",
    ("yesno", "ar"): "\n\nيمكنك التفكير قبل الإجابة. يجب أن يكون السطر الأخير من ردك 'Answer: Yes' أو 'Answer: No'.",
}


def opts_text(names):
    return " ".join(f"({i + 1}) {n}" for i, n in enumerate(names)).rstrip(".")


# ----------------------------------------------------------------------------- generator
def generate(repo, per_task_city=10):
    rng = random.Random(SEED)
    D = json.load(open(f"{repo}/json/districts.json", encoding="utf-8"))
    C = json.load(open(f"{repo}/json/cities.json", encoding="utf-8"))
    city_name = {str(c["city_id"]): (c["name_en"], c["name_ar"]) for c in C}
    dist_by_city = {}
    for d in D:
        dist_by_city.setdefault(str(d["city_id"]), []).append(d)
    # national name index for the CITY task (which cities contain a district with this name)
    name_cities = {}
    for d in D:
        for key in (d["name_ar"].strip(), d["name_en"].strip().lower()):
            name_cities.setdefault(key, set()).add(str(d["city_id"]))
    # comparable distractors: other large cities (>= 40 districts), so city size is not a cue
    big_cities = [cid for cid, ds in dist_by_city.items() if len(ds) >= 40]

    items = []
    for cid, (cen, car) in CITIES.items():
        ds = []
        for d in dist_by_city[cid]:
            g = to_poly(d["boundaries"])
            if g is None:
                continue
            ds.append({"en": d["name_en"].strip(), "ar": d["name_ar"].strip(), "ll": g})
        # keep names unique within the city
        cnt_en, cnt_ar = Counter(x["en"] for x in ds), Counter(x["ar"] for x in ds)
        ds = [x for x in ds if cnt_en[x["en"]] == 1 and cnt_ar[x["ar"]] == 1]
        lat0 = sum(x["ll"].centroid.y for x in ds) / len(ds)
        proj = Proj(lat0)
        for x in ds:
            x["m"] = proj.fwd(x["ll"]); x["c"] = x["m"].centroid; x["cll"] = x["ll"].centroid
        ds = [x for x in ds if x["m"].area >= 250_000]          # >= 0.25 km^2
        rng.shuffle(ds)

        def nearest(x, k):
            return sorted((y for y in ds if y is not x), key=lambda y: x["c"].distance(y["c"]))[:k]

        def add(task, k, gold, q_en, q_ar, link=None):
            dim, typ = TASKS[task]
            for lang, q in (("en", q_en), ("ar", q_ar)):
                items.append({"id": f"ksa-{task}-{cen}-{k}-{lang}", "pair": f"ksa-{task}-{cen}-{k}",
                              "link": link, "task": task, "dim": dim, "type": typ, "city": cen,
                              "lang": lang, "q": q + INSTR[(typ, lang)], "gold": gold})

        # ---- PIDK and PIDC: point -> district
        made_k = made_c = 0
        for xi, x in enumerate(ds):
            if made_k >= per_task_city and made_c >= per_task_city:
                break
            inner = x["m"].buffer(-150)
            if inner.is_empty:
                continue
            minx, miny, maxx, maxy = inner.bounds
            pt = None
            for _ in range(200):
                p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
                if inner.contains(p):
                    pt = p; break
            if pt is None:
                continue
            ll = proj.inv(pt)
            cand = [x] + nearest(x, 4)
            order = list(range(5)); rng.shuffle(order)
            opts = [cand[i] for i in order]
            gold = str(order.index(0) + 1)
            loc_en = f"latitude {ll.y:.5f}, longitude {ll.x:.5f}"
            loc_ar = f"خط العرض {ll.y:.5f}، خط الطول {ll.x:.5f}"
            if made_k < per_task_city:
                add("PIDK", made_k, gold,
                    f"Question: A location in {cen}, Saudi Arabia, has coordinates ({loc_en}). Which district (neighborhood) is it in? Options: {opts_text([o['en'] for o in opts])}.",
                    f"سؤال: يقع موقع في مدينة {car} بالمملكة العربية السعودية عند الإحداثيات ({loc_ar}). في أي حي يقع هذا الموقع؟ الخيارات: {opts_text([o['ar'] for o in opts])}.",
                    link=f"pid-{cen}-{xi}")
                made_k += 1
            if made_c < per_task_city:
                simp = [simplify_for_prompt(o["m"], proj) for o in opts]
                inside = [sm.contains(pt) for sm, _ in simp]
                if inside.count(True) == 1 and inside.index(True) == order.index(0) \
                        and simp[order.index(0)][0].buffer(-40).contains(pt):
                    b_en = "\n".join(f"District {i + 1} ({o['en']}): {fmt_poly(sll)}" for i, (o, (_, sll)) in enumerate(zip(opts, simp)))
                    b_ar = "\n".join(f"الحي {i + 1} ({o['ar']}): {fmt_poly(sll)}" for i, (o, (_, sll)) in enumerate(zip(opts, simp)))
                    add("PIDC", made_c, gold,
                        "Question: Below are the boundaries of several districts in " + cen + ", each given as a list of (latitude, longitude) vertices:\n"
                        + b_en + f"\nWhich district contains the point ({loc_en})? Options: {opts_text([f'District {i + 1}' for i in range(5)])}.",
                        "سؤال: فيما يلي حدود عدة أحياء في مدينة " + car + "، كل منها قائمة رؤوس بصيغة (خط العرض، خط الطول):\n"
                        + b_ar + f"\nأي حي يحتوي النقطة ({loc_ar})؟ الخيارات: {opts_text([f'الحي {i + 1}' for i in range(5)])}.",
                        link=f"pid-{cen}-{xi}")
                    made_c += 1

        # ---- DIRK: direction from A to B (centroids), unambiguous sectors only
        made, tries = 0, 0
        while made < per_task_city and tries < 20000:
            tries += 1
            a, b = rng.sample(ds, 2)
            dist = a["c"].distance(b["c"])
            if not 3000 <= dist <= 25000:
                continue
            brg = bearing(a["cll"].x, a["cll"].y, b["cll"].x, b["cll"].y)
            sec = int(round(brg / 45.0)) % 8
            if abs(((brg - sec * 45 + 180) % 360) - 180) > 12.5:
                continue
            add("DIRK", made, str(sec + 1),
                f"Question: In {cen}, Saudi Arabia, in which direction is the center of {b['en']} from the center of {a['en']}? Options: {opts_text(DIRS_EN)}.",
                f"سؤال: في مدينة {car}، في أي اتجاه يقع مركز {b['ar']} بالنسبة إلى مركز {a['ar']}؟ الخيارات: {opts_text(DIRS_AR)}.")
            made += 1

        # ---- NEARK: which candidate is nearest to X (clear margin)
        made = 0
        for x in ds:
            if made >= per_task_city:
                break
            ranked = sorted((y for y in ds if y is not x), key=lambda y: x["c"].distance(y["c"]))
            if len(ranked) < 30:
                continue
            near = ranked[0]
            others = rng.sample(ranked[4:30], 3)
            if x["c"].distance(near["c"]) > 0.6 * min(x["c"].distance(o["c"]) for o in others):
                continue
            cand = [near] + others
            order = list(range(4)); rng.shuffle(order)
            opts = [cand[i] for i in order]
            add("NEARK", made, str(order.index(0) + 1),
                f"Question: In {cen}, Saudi Arabia, which of the following districts has its center closest to the center of {x['en']}? Options: {opts_text([o['en'] for o in opts])}.",
                f"سؤال: في مدينة {car}، أي الأحياء التالية مركزه هو الأقرب إلى مركز {x['ar']}؟ الخيارات: {opts_text([o['ar'] for o in opts])}.")
            made += 1

        # ---- ADJK: adjacency, balanced yes/no
        yes = no = 0; tries = 0
        while (yes < per_task_city // 2 or no < per_task_city - per_task_city // 2) and tries < 50000:
            tries += 1
            a, b = rng.sample(ds, 2)
            gap = a["m"].distance(b["m"])
            shared = a["m"].buffer(30).intersection(b["m"]).area
            if gap <= 30 and shared > 2000 and yes < per_task_city // 2:
                gold = "yes"; yes += 1
            elif 1000 <= gap <= 6000 and no < per_task_city - per_task_city // 2:
                gold = "no"; no += 1
            else:
                continue
            add("ADJK", yes + no - 1, gold,
                f"Question: In {cen}, Saudi Arabia, do the districts {a['en']} and {b['en']} share a border?",
                f"سؤال: في مدينة {car}، هل يشترك {a['ar']} و{b['ar']} في حدود مشتركة (متجاوران)؟")

        # ---- CITYK: district -> city, answer unique among options
        made = 0
        for x in ds:
            if made >= per_task_city:
                break
            pool = [c for c in big_cities if c != cid and c not in name_cities.get(x["ar"], set())
                    and c not in name_cities.get(x["en"].lower(), set())]
            if len(pool) < 4:
                continue
            cand = [cid] + rng.sample(pool, 4)
            order = list(range(5)); rng.shuffle(order)
            opts = [cand[i] for i in order]
            add("CITYK", made, str(order.index(0) + 1),
                f"Question: {x['en']} is a district in which Saudi city? Options: {opts_text([city_name[c][0] for c in opts])}.",
                f"سؤال: {x['ar']} حيٌّ في أي مدينة سعودية؟ الخيارات: {opts_text([city_name[c][1] for c in opts])}.")
            made += 1
    return items


if __name__ == "__main__":
    repo, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    items = generate(repo, n)
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(len(items), "items", Counter((i["task"], i["lang"]) for i in items))
