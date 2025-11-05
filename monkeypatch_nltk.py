# Patch sem recursão para alinhar _regexs e _regexps no NLTK RegexpTagger
from nltk.tag.sequential import RegexpTagger

if not getattr(RegexpTagger, "_patched_regexps_property", False):
    def _get_regexps(self):
        d = object.__getattribute__(self, "__dict__")
        if "_regexps" in d:
            return d["_regexps"]
        if "_regexs" in d:
            return d["_regexs"]
        return []

    def _set_regexps(self, value):
        d = object.__getattribute__(self, "__dict__")
        d["_regexps"] = value
        d["_regexs"] = value

    try:
        RegexpTagger._regexps = property(_get_regexps, _set_regexps)
    except Exception:
        _orig = RegexpTagger.__init__
        def _init(self, *a, **k):
            _orig(self, *a, **k)
            d = object.__getattribute__(self, "__dict__")
            if "_regexs" in d and "_regexps" not in d:
                d["_regexps"] = d["_regexs"]
        RegexpTagger.__init__ = _init

    RegexpTagger._patched_regexps_property = True