"""Embeds outputs/tables/final_model_M2.json into the calculator template -> dst/mbc_pfs_calculator.html."""
import json
from common import OUT_TAB, ROOT
t = (ROOT / "dst" / "pfs_calculator_template.html").read_text(encoding="utf8")
m = json.load(open(OUT_TAB / "final_model_M2.json"))
(ROOT / "dst" / "mbc_pfs_calculator.html").write_text(t.replace("__MODEL__", json.dumps(m)).replace("__MEDIANS__", json.dumps(m["medians"])), encoding="utf8")
print("calculator written")
