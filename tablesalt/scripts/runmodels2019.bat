

::REMEMBER SOLO_ZONE_PRIS in resources/config/config_revenue.ini!!!!!!!!!!!!
call conda activate tablesalt

python salesdatamerge.py -y 2019 -d H:\revenue\inputdata\2019\salesdata
python delrejsersetup.py -y 2019 -i H:\Data\Rejsedata\2019\Delrejser -o H:\datastores -c 1000000
python calculateshares.py -y 2019
python pendlersetup.py -y 2019 -z H:\revenue\inputdata\2019\new_zoner.csv -p H:\revenue\inputdata\2019\new_products.csv
python pendlerkeys.py -y 2019 -z H:\revenue\inputdata\2019\new_zoner.csv -p H:\revenue\inputdata\2019\new_products.csv
python singlekeys.py -y 2019 -r 0
python singlekeys.py -y 2019 -r 1
python singlekeys.py -y 2019 -r 2
python subtakst.py -y 2019 -r 0
python salesoutput.py -y 2019 -m 1 & python salesoutput.py -y 2019 -m 2 & python salesoutput.py -y 2019 -m 3 
python salesoutput.py -y 2019 -m 4 & python salesoutput.py -y 2019 -m 5 & python salesoutput.py -y 2019 -m 6
python kombicardshares.py -y 2019 -z H:\revenue\inputdata\2019\new_zoner.csv -p H:\revenue\inputdata\2019\new_products.csv
