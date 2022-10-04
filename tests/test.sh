# if [[ -f "iSPEC.db" ]];
# then
#     rm "iSPEC.db"
# fi

ROOTDIR=".."

mspcmonitor db create --force
mspcmonitor db add-experiments-table "$ROOTDIR"/data/20221904_ispec_cur_Experiments_metadata_short.csv
mspcmonitor db add-experimentruns-table "$ROOTDIR"/data/20222404_ispec_cur_ExperimentRuns_metadata_export_short.csv
mspcmonitor db add-genes-table ./testdata/genetable_short100.tsv

mspcmonitor db add-e2g-table ./testdata/99995_426_6_labelnone_e2g_QUAL_short.tsv testdata/99995_426_6_labelnone_e2g_QUANT_short.tsv
mspcmonitor db add-psm-table ./testdata/99995_426_6_labelnone_psms_QUAL_short.tsv testdata/99995_426_6_labelnone_0_psms_QUANT_short.tsv

