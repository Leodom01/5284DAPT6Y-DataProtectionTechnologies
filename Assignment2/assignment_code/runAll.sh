python3 xesToCsv.py
echo "All dataset converted in CSV"
python3 ../PRETSA/add_annotation_duration.py sepsis "../PRETSA/datasets/Sepsis cases/sepsis.csv"
python3 ../PRETSA/add_annotation_duration.py traffic "../PRETSA/datasets/Road traffic management dataset/traffic.csv"
python3 ../PRETSA/add_annotation_duration.py environment "../PRETSA/datasets/Environmental permit application/environmental.csv"
echo "Added Duration to all datasets"