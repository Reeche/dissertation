from condor_utils import submit_sub_file

bid = 25
script = '1_create_df.py'  # The file that you want to run on the cluster.


submit_sub_file("sub_multiple_analysis_job.sub", bid)
