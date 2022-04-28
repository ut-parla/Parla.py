set -v
python qr.py --rows  500000 --cols 1000 --ngpus 1 --precision single
echo
python qr.py --rows 1000000 --cols 1000 --ngpus 1 --precision single
echo
python qr.py --rows  500000 --cols 1000 --ngpus 1 --precision double
echo
python qr.py --rows 1000000 --cols 1000 --ngpus 1 --precision double
echo
python qr.py --rows  500000 --cols 1000 --ngpus 2 --precision single
echo
python qr.py --rows 1000000 --cols 1000 --ngpus 2 --precision single
echo
python qr.py --rows  500000 --cols 1000 --ngpus 2 --precision double
echo
python qr.py --rows 1000000 --cols 1000 --ngpus 2 --precision double
echo
