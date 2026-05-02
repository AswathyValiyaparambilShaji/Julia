#!/bin/bash


BASE=/home3/avaliyap/Documents/Julia_new/Julia/Codes/Budget
PBS=/home3/avaliyap/Documents/Julia_new/Julia/m_run_j.pbs


# Submit all jobs chained one after another
JOB1=$(qsub -v JULIA_SCRIPT=$BASE/NIW/UVW_bpfilter.jl $PBS)
echo "Job 1 submitted: $JOB1"


JOB2=$(qsub -v JULIA_SCRIPT=$BASE/NIW/buoyancy_pd_niw.jl -W depend=afterok:$JOB1 $PBS)
echo "Job 2 submitted: $JOB2"


JOB3=$(qsub -v JULIA_SCRIPT=$BASE/NIW/Shear_pdv_niw.jl -W depend=afterok:$JOB2 $PBS)
echo "Job 3 submitted: $JOB3"

JOB4=$(qsub -v JULIA_SCRIPT=$BASE/NT/UVW_nt_filtered.jl -W depend=afterok:$JOB3 $PBS)
echo "Job 4 submitted: $JOB4"

#JOB5=$(qsub -v JULIA_SCRIPT=$BASE/NT/UVW_nt_filtered.jl -W depend=afterok:$JOB4 $PBS)
#echo "Job 5 submitted: $JOB5"

# Add as many scripts as you need following the same pattern...




