call conda activate tablesalt
python delrejsersetup.py -y 2019 -i H:\Data\Rejsedata\2019\Delrejser -o H:\datastores
python calculateshares.py -y 2019
python pendlersetup.py -y 2019 -z H:\revenue\inputdata\2019\new_zoner.csv -p H:\revenue\inputdata\2019\new_products.csv
python pendlerkeys.py -y 2019 -z H:\revenue\inputdata\2019\new_zoner.csv -p H:\revenue\inputdata\2019\new_products.csv
python singlekeys.py -y 2019
python subtakst.py -y 2019 -r 0
pause