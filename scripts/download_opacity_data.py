"""Download exo_k-readable opacity data (ExoMol k-tables / cross sections, HITRAN CIA, MT_CKD H2O continuum) into
src/two_column/opacity_data.

    python scripts/download_opacity_data.py                  (k-tables for CO2, H2O, CO + CIA + H2O continuum, ~1.3 GB)
    python scripts/download_opacity_data.py --xsec           (also the R15000 TauREx cross sections, ~0.4 GB each)
    python scripts/download_opacity_data.py --only cia       (just the CIA and continuum tables, ~175 MB)
    python scripts/download_opacity_data.py --list           (print what would be fetched, download nothing)

Sources, from https://lab.pages.in2p3.fr/whiplash-public/exo_k/where_to_find_data.html:
  ExoMol (Chubb et al. 2021, A&A 646, A21): R1000 k-tables (0.3-50 um) in petitRADTRANS .h5 format, read with exo_k.Ktable.
  HITRAN CIA (Karman et al. 2019): .cia files, read with exo_k.Cia_table(filename=...).
  MT_CKD 4.3 (Mlawer et al. 2012, 2023; github.com/AER-RC/MT_CKD_H2O): H2O self and foreign continuum.
H2 has no ExoMol line list (its quadrupole lines are negligible), so H2 enters only through CIA. HITRAN has no CO CIA.

Layout: H2O/, CO2/, CO/ hold k-tables; cia/ holds one table per molecule pair (exo_k's CIAdatabase keeps only one per pair),
all loadable with exo_k.Cia_table; source/ holds raw files that cia/ tables are built from (not for direct use). Two cia/
products are built here rather than downloaded:
  CO2-CO2_2024.h5          The HITRAN 2024 CO2-CO2 file holds ten blocks from several studies over five bands, and exo_k only
                           reads the first block. The bands are merged into one table (see build_co2_co2_cia).
  H2O-H2O_MT_CKD_4.3.h5,   MT_CKD self and foreign (air, stored as N2) continua as CIA coefficients. MT_CKD assumes line
  H2O-N2_MT_CKD_4.3.h5     opacity is cut 25 cm-1 from line centre; the ExoMol tables cut at <= 25 cm-1, so it does not double
                           count. The self continuum's temperature power law is only constrained near 250-350 K.
exo_k holds temperatures outside a table's grid at the nearest edge, and interpolates in log space, so built tables have
non-positive values replaced by POSITIVE_FLOOR.
"""
import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "src" / "two_column" / "opacity_data"
CIA_DIR = DATA_DIR / "cia"
SOURCE_DIR = DATA_DIR / "source"
EXOMOL = "https://www.exomol.com/db"
HITRAN_CIA = "https://hitran.org/data/CIA"
MT_CKD_VERSION = "4.3"
MT_CKD_URL = f"https://raw.githubusercontent.com/AER-RC/MT_CKD_H2O/{MT_CKD_VERSION}/data/absco-ref_wv-mt-ckd.nc"

# Main isotopologue and recommended line list for each molecule: (database path, file stem).
LINE_LISTS = {
    "H2O": ("H2O/1H2-16O/POKAZATEL", "1H2-16O__POKAZATEL__"),
    "CO2": ("CO2/12C-16O2/UCL-4000", "12C-16O2__UCL-4000."),
    "CO": ("CO/12C-16O/Li2015", "12C-16O__Li2015."),
}
KTABLE_SUFFIX = "R1000_0.3-50mu.ktable.petitRADTRANS.h5"
XSEC_SUFFIX = "R15000_0.3-50mu.xsec.TauREx.h5"

# HITRAN "main" CIA pairs among CO2, H2O, H2 (and He, H2's usual partner) that exo_k reads as they are. The 2011 H2-H2 file is
# used because the 2018 one labels the pair "n-H2 -- n-H2", which exo_k cannot parse without hand editing.
CIA_FILES = ["H2-H2_2011.cia", "H2-He_2011.cia", "CO2-H2_2024.cia", "CO2-H2O_2024.cia"]
CO2_CO2_FILE = "CO2-CO2_2024.cia"

# CO2-CO2 bands (cm-1); a block belongs to the band containing its mid wavenumber. The lone 298 K block at 2850-3250 cm-1 (a
# different study) is ~30% above the 295 K block of the 230-295 K set, so merging it would put a jump in T; it is left out.
CO2_CO2_BANDS = [(1, 750), (1120, 1850), (2510, 2850), (2850, 3250), (3995, 4500)]
CO2_CO2_SKIP = {(2850, 298.0)}    # (band lower edge, block temperature)
BAND_EDGE_STEP = 0.01             # cm^-1 beyond each band edge where the merged CO2-CO2 table drops to POSITIVE_FLOOR

MT_CKD_TGRID = np.arange(200., 701., 25.)    # same span as the LMDZ generic-model CKD tables
POSITIVE_FLOOR = 1e-60                       # cm^5 molecule^-2, ~15 orders below any real CIA value here
BOLTZ_CGS = 1.380649e-16                     # erg/K
C2 = 1.4387769                               # second radiation constant hc/k, cm K


def file_list(only, xsec):
    """(url, destination) pairs for the requested data. only: "all", "lines" or "cia"."""
    files = []
    if only in ("all", "lines"):
        suffixes = [KTABLE_SUFFIX] + ([XSEC_SUFFIX] if xsec else [])
        for molecule, (path, stem) in LINE_LISTS.items():
            files += [(f"{EXOMOL}/{path}/{stem}{s}", DATA_DIR / molecule / f"{stem}{s}") for s in suffixes]
    if only in ("all", "cia"):
        files += [(f"{HITRAN_CIA}/main/{name}", CIA_DIR / name) for name in CIA_FILES]
        files += [(f"{HITRAN_CIA}/main/{CO2_CO2_FILE}", SOURCE_DIR / CO2_CO2_FILE),
                  (MT_CKD_URL, SOURCE_DIR / f"absco-ref_wv-mt-ckd_{MT_CKD_VERSION}.nc")]
    return files


def download(url, dest, force=False):
    """Stream `url` to `dest` via a .part file, so an interrupted download is never mistaken for a finished one."""
    if dest.exists() and not force:
        print(f"exists   {dest.relative_to(REPO_ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "exo-ocean opacity downloader"})
    with urllib.request.urlopen(request) as response, open(part, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total and done % (16 << 20) < len(chunk):    # every ~16 MB
                print(f"\r{100 * done / total:5.1f}%  {done / 1e6:7.1f}/{total / 1e6:.1f} MB  {dest.name}", end="", flush=True)
    part.replace(dest)
    print(f"\rdone     {dest.relative_to(REPO_ROOT)} ({done / 1e6:.1f} MB)" + " " * 20)


def write_cia_table(path, pair, wns, tgrid, abs_coeff):
    """Write an exo_k hdf5 CIA table. abs_coeff: (len(tgrid), len(wns)) in cm^5 molecule^-2, floored at POSITIVE_FLOOR."""
    import exo_k as xk
    table = xk.Cia_table()
    table.mol1, table.mol2 = pair
    table.wns, table.tgrid = np.asarray(wns, float), np.asarray(tgrid, float)
    table.abs_coeff = np.maximum(abs_coeff, POSITIVE_FLOOR)
    table.abs_coeff_unit = "cm^5"
    table.Nw, table.Nt = table.wns.size, table.tgrid.size
    table.write_hdf5(str(path))
    print(f"built    {path.relative_to(REPO_ROOT)} ({table.Nt} T x {table.Nw} wn, {table.tgrid[0]:g}-{table.tgrid[-1]:g} K)")


def read_hitran_blocks(path):
    """All (temperature, wavenumbers, coefficients) blocks of a HITRAN .cia file, whatever their wavenumber ranges."""
    lines = Path(path).read_text().splitlines()
    blocks, i = [], 0
    while i < len(lines):
        header = lines[i].split()
        if not header:
            i += 1
            continue
        n, temperature = int(header[3]), float(header[4])
        data = np.array([row.split()[:2] for row in lines[i + 1:i + 1 + n]], float)
        blocks.append((temperature, data[:, 0], data[:, 1]))
        i += n + 1
    return blocks


def build_co2_co2_cia(force=False):
    """Merge the multi-band HITRAN CO2-CO2 file (bands in CO2_CO2_BANDS) into one exo_k table, CO2-CO2_2024.h5.

    Within a band, blocks come from different studies on different grids; they are interpolated onto the finest block's
    grid over the wavenumbers every block covers. The bands are then put on the union of their temperatures, each band
    interpolated linearly in log(k) and held at its own edge temperatures, which is what exo_k does when interpolating a
    single band's table. Points BAND_EDGE_STEP outside each band are set to POSITIVE_FLOOR, so that the linear wavenumber
    interpolation in exo_k's sample() does not fill the gaps between bands.
    """
    path = CIA_DIR / "CO2-CO2_2024.h5"
    if path.exists() and not force:
        print(f"exists   {path.relative_to(REPO_ROOT)}")
        return
    blocks = read_hitran_blocks(SOURCE_DIR / CO2_CO2_FILE)
    bands = []    # (temperatures, wavenumbers, coefficients) per band
    for lo, hi in CO2_CO2_BANDS:
        band = [b for b in blocks if lo <= 0.5 * (b[1][0] + b[1][-1]) <= hi and (lo, b[0]) not in CO2_CO2_SKIP]
        band.sort(key=lambda b: b[0])
        tgrid = np.array([b[0] for b in band])
        if np.unique(tgrid).size != tgrid.size:
            raise ValueError(f"duplicate temperatures in CO2-CO2 band {lo}-{hi} cm-1: {tgrid}")
        w_min, w_max = max(b[1][0] for b in band), min(b[1][-1] for b in band)
        finest = min(band, key=lambda b: np.median(np.diff(b[1])))[1]
        wns = finest[(finest >= w_min) & (finest <= w_max)]
        k = np.maximum(np.array([np.interp(wns, w, kb) for _, w, kb in band]), POSITIVE_FLOOR)
        bands.append((tgrid, wns, k))

    tgrid = np.unique(np.concatenate([b[0] for b in bands]))
    wns_parts, k_parts = [], []
    for t_band, w_band, k_band in bands:
        log_k = np.array([[np.interp(t, t_band, col) for t in tgrid] for col in np.log(k_band).T]).T   # (Nt, Nw)
        floor = np.full((tgrid.size, 1), POSITIVE_FLOOR)
        wns_parts.append(np.concatenate(([w_band[0] - BAND_EDGE_STEP], w_band, [w_band[-1] + BAND_EDGE_STEP])))
        k_parts.append(np.hstack((floor, np.exp(log_k), floor)))
    write_cia_table(path, ("CO2", "CO2"), np.concatenate(wns_parts), tgrid, np.hstack(k_parts))


def build_mt_ckd(force=False):
    """H2O-H2O and H2O-N2 CIA tables from the MT_CKD reference coefficients, on MT_CKD_TGRID.

    MT_CKD (mt_ckd_h2o_module.f90) gives the absorption coefficient alpha = n_w R(nu, T) C(nu, T) n_partner / n_ref, with
    R = nu tanh(c2 nu / 2T), n_ref the number density at ref_press and ref_temp, and C_self = C_self_ref (T_ref/T)^texp,
    C_foreign independent of T. So the CIA coefficient (alpha = k n_w n_partner) is k = R C / n_ref.
    """
    import netCDF4
    products = {"self": CIA_DIR / f"H2O-H2O_MT_CKD_{MT_CKD_VERSION}.h5", "foreign": CIA_DIR / f"H2O-N2_MT_CKD_{MT_CKD_VERSION}.h5"}
    if all(p.exists() for p in products.values()) and not force:
        for p in products.values():
            print(f"exists   {p.relative_to(REPO_ROOT)}")
        return
    with netCDF4.Dataset(SOURCE_DIR / f"absco-ref_wv-mt-ckd_{MT_CKD_VERSION}.nc") as ds:
        wns = ds["wavenumbers"][:].data
        self_ref, foreign_ref, texp = (ds[v][:].data for v in ("self_absco_ref", "for_absco_ref", "self_texp"))
        p_ref, t_ref = float(ds["ref_press"][:]), float(ds["ref_temp"][:])
    keep = wns > 0                                       # the grid starts at -20 cm-1 for the LBLRTM interpolation
    wns, self_ref, foreign_ref, texp = wns[keep], self_ref[keep], foreign_ref[keep], texp[keep]
    n_ref = p_ref * 1e3 / (BOLTZ_CGS * t_ref)            # mbar -> dyn cm^-2 -> cm^-3
    t = MT_CKD_TGRID[:, None]
    radiation = wns * np.tanh(C2 * wns / (2 * t))
    write_cia_table(products["self"], ("H2O", "H2O"), wns, MT_CKD_TGRID, radiation * self_ref * (t_ref / t) ** texp / n_ref)
    write_cia_table(products["foreign"], ("H2O", "N2"), wns, MT_CKD_TGRID, radiation * foreign_ref / n_ref)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["all", "lines", "cia"], default="all",
                        help="line opacities, CIA + continuum, or both (default)")
    parser.add_argument("--xsec", action="store_true", help="also fetch the R15000 TauREx cross-section tables")
    parser.add_argument("--force", action="store_true", help="re-download and rebuild files that already exist")
    parser.add_argument("--list", action="store_true", help="print the files and exit")
    args = parser.parse_args()

    files = file_list(args.only, args.xsec)
    if args.list:
        for url, dest in files:
            print(f"{url}\n    -> {dest.relative_to(REPO_ROOT)}")
        return
    failed = []
    for url, dest in files:
        try:
            download(url, dest, args.force)
        except OSError as err:
            print(f"\nfailed   {url}: {err}", file=sys.stderr)
            failed.append(url)
    if failed:
        sys.exit(f"{len(failed)} download(s) failed")
    if args.only in ("all", "cia"):
        build_co2_co2_cia(args.force)
        build_mt_ckd(args.force)


if __name__ == "__main__":
    main()
