"""One small CPU check: frozen inequality, paired counts, and ZIP range decoding."""
import io
from pathlib import Path
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.medtrace import stage16 as a, stage16_sources as s


def test_contract():
    route = dict(activated=True, nearest_distance=2., radius=10.)
    assert a.route_values(route) == (2., 10., .2, .8)
    assert a.accepted(route, .8) and not a.accepted(route, a.math.nextafter(.8, a.math.inf))
    assert a.exact_p(4,0) == .125 and a.exact_p(0,0) == 1
    assert a.auroc([1.,.5], [.5,0.]) == .875
    r = dict(edit=1, metric='I_Locality', base_exact=False, forced_exact=True,
        base_semantic=False, forced_semantic=True, forced_preserve=False,
        base_target_copy=False, forced_target_copy=True, base_cap=False, forced_cap=False)
    off = a.summarize([r], [False], 'semantic'); on = a.summarize([r], [True], 'semantic')
    assert off['base_wrong_correction'] == 0 and on['base_wrong_correction'] == 1
    assert a.summarize([r], [False], 'exact')['total_success'] == 1
    assert a.summarize([r], [True], 'exact')['total_success'] == 0
    data = io.BytesIO()
    with zipfile.ZipFile(data,'w',zipfile.ZIP_DEFLATED) as z: z.writestr('images/example.jpg',b'image payload')
    with zipfile.ZipFile(io.BytesIO(data.getvalue())) as z: info = z.infolist()[0]
    original = s.RemoteZip
    class LocalRange(io.BytesIO):
        def __init__(self, *args): super().__init__(data.getvalue()); self.bytes_read=0
        def read(self,n=-1):
            result=super().read(n); self.bytes_read+=len(result); return result
    try:
        s.RemoteZip=LocalRange
        assert s.member_bytes('unused',len(data.getvalue()),info)[0] == b'image payload'
    finally: s.RemoteZip=original


if __name__ == '__main__':
    test_contract(); print('Stage16 CPU contract passed')
