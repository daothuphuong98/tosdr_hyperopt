cd /home/ubuntu/tosdr
export PYTHONPATH=.
python3 hyperopt_xgb_cat_lgbm.py >> log/hyperopt.log 2>&1 &