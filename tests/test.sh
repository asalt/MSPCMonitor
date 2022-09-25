ROOTDIR=".."
mspcmonitor db create --force
mspcmonitor db add-experiments-table "$ROOTDIR"/data/20221904_ispec_cur_Experiments_metadata_short.csv
mspcmonitor db add-expruns-table "$ROOTDIR"/data/20222404_ispec_cur_ExperimentRuns_metadata_export_short.csv


if [[ -f "iSPEC.db" ]];
then
    rm "iSPEC.db"
fi

