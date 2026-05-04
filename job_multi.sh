#!/bin/bash


BASE=/home3/avaliyap/Documents/Julia_new/Julia/Codes/Budget
PBS=/home3/avaliyap/Documents/Julia_new/Julia/m_run_j.pbs


# Submit all jobs chained one after another
JOB1=$(qsub -v JULIA_SCRIPT=$BASE/NT/Ke_b_nt.jl  $PBS)
echo "Job 1 submitted: $JOB1"

JOB2=$(qsub -v JULIA_SCRIPT=$BASE/NT/FlxDiv_plot_nt.jl  -W depend=afterok:$JOB1 $PBS)
echo "Job 2 submitted: $JOB2"

JOB3=$(qsub -v JULIA_SCRIPT=$BASE/NT/APE_nt.jl  -W depend=afterok:$JOB2 $PBS)
echo "Job 3 submitted: $JOB3"

JOB4=$(qsub -v JULIA_SCRIPT=$BASE/NT/APE_plot_nt.jl  -W depend=afterok:$JOB3 $PBS)
echo "Job 4 submitted: $JOB4"

JOB5=$(qsub -v JULIA_SCRIPT=$BASE/NT/ADV_PE_nt_plot.jl  -W depend=afterok:$JOB4 $PBS)
echo "Job 5 submitted: $JOB5"

JOB6=$(qsub -v JULIA_SCRIPT=$BASE/NT/ADV_PE_nt_plot.jl -W depend=afterok:$JOB5 $PBS)
echo "Job 6 submitted: $JOB6"

JOB7=$(qsub -v JULIA_SCRIPT=$BASE/NT/boyancy_pd_nt.jl -W depend=afterok:$JOB6 $PBS)
echo "Job 7 submitted: $JOB7"

JOB8=$(qsub -v JULIA_SCRIPT=$BASE/NT/KE_plot_nt.jl -W depend=afterok:$JOB7 $PBS)
echo "Job 8 submitted: $JOB8"

JOB9=$(qsub -v JULIA_SCRIPT=$BASE/NT/Buoyancy_pd_nt_plot.jl -W depend=afterok:$JOB8 $PBS)
echo "Job 9 submitted: $JOB9"

#JOB10=$(qsub -v JULIA_SCRIPT=$BASE/NT/Tendency_term_nt.jl -W depend=afterok:$JOB9 $PBS)
#echo "Job 10 submitted: $JOB10"

#JOB5=$(qsub -v JULIA_SCRIPT=$BASE -W depend=afterok:$JOB4 $PBS)
#echo "Job 5 submitted: $JOB5"
# Add as many scripts as you need following the same pattern...




