
::REMEMBER SOLO_ZONE_PRIS in resources/config/config_revenue.ini!!!!!!!!!!!!

call conda activate tablesalt
python salesdatamerge.py -y 2020 -d H:\revenue\inputdata\2020\salesdata 
python delrejsersetup.py -y 2020 -i H:\Data\Rejsedata\2020\Delrejser -o H:\datastores -c 1000000 
python calculateshares.py -y 2020 
python pendlersetup.py -y 2020 -z H:\revenue\inputdata\2020\RK_SeasonPass_2020_Zones.csv -p H:\revenue\inputdata\2020\RK_SeasonPass_2020_Sales.csv
python pendlerkeys.py -y 2020 -z H:\revenue\inputdata\2020\RK_SeasonPass_2020_Zones.csv -p H:\revenue\inputdata\2020\RK_SeasonPass_2020_Sales.csv
python singlekeys.py -y 2020 -r 0
python singlekeys.py -y 2020 -r 1
python singlekeys.py -y 2020 -r 2
python subtakst.py -y 2020 -r 0
python salesoutput.py -y 2020 -m 1 & python salesoutput.py -y 2020 -m 2 & python salesoutput.py -y 2020 -m 3 
python salesoutput.py -y 2020 -m 4 & python salesoutput.py -y 2020 -m 5 & python salesoutput.py -y 2020 -m 6
python kombicardshares.py -y 2020 -z H:\revenue\inputdata\2020\RK_SeasonPass_2020_Sales.csv -p H:\revenue\inputdata\2020\RK_SeasonPass_2020_Sales.csv
