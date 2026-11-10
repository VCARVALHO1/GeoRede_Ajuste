"""Run the documented GeoRede article profile."""
from pathlib import Path
import argparse,subprocess,sys,os
import importlib.util
root=Path(__file__).resolve().parent
# Prefer the project's environment even when Windows associates .py with 3.9.
local_python=root/'.venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
if local_python.is_file() and Path(sys.prefix).resolve() != (root/'.venv').resolve():
    raise SystemExit(subprocess.call([str(local_python),str(Path(__file__).resolve()),*sys.argv[1:]]))
if sys.version_info < (3,12):
    raise SystemExit('GeoRede requires Python 3.12 or newer. Create .venv with Python 3.12 and install requirements.txt.')
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data',type=Path,default=root/'data',help='Station XYZ input folder')
p.add_argument('--out',type=Path,default=root/'output/run_01',help='New output folder; existing folders are never overwritten')
args=p.parse_args()
missing=[name for name in ('numpy','pandas','scipy','matplotlib') if importlib.util.find_spec(name) is None]
if missing:
    p.error('Missing dependencies: '+', '.join(missing)+'. Run: "'+sys.executable+'" -m pip install -r "'+str(root/'requirements.txt')+'"')
if not args.data.is_dir():p.error('Input folder does not exist')
if args.out.exists():p.error('Output already exists. Choose another --out folder.')
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
env.setdefault('MPLCONFIGDIR',str(root/'.cache'/'matplotlib'))
env.setdefault('PYTHONIOENCODING','utf-8')
cmd=[sys.executable,str(root/'src/runner.py'),'--source-root',str(root),'--data',str(args.data.resolve()),'--out',str(args.out.resolve()),'--profile','network_baarda_mad','--max-iter','50','--station-max-iter','25','--network-tol','1e-8','--linear-solver','adaptive_lm','--sigma-mm','2','--power','0.8','--critical-value','3','--alpha-global','0.05','--max-cycles','20','--network-geometry','coupled']
raise SystemExit(subprocess.call(cmd,cwd=root,env=env))
