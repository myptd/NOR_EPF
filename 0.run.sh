
## install dependencies

## run only once
# conda env create -f env.yml

conda env update -f env.yml

## activate environment
conda activate elec

## collect weather data
python bin/collect_weather_openmeteo.py

## collect reservoir data
python bin/collect_reservoir_nve.py

## collect 

python bin/collect_commodities_yfinance.py

## get ENTSOE data
export ENTSOE_API_KEY="f9895a63-9cb2-42af-9fd7-a54c07153afe"

python bin/collect_entsoe.py

## data preprocessing
python bin/data_preprocess.py --no-csv