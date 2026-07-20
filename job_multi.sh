
#!/bin/bash


BASE=/home3/avaliyap/Documents/Julia_new/Julia/Codes/
PBS=/home3/avaliyap/Documents/Julia_new/Julia/m_run_j.pbs


# Submit all jobs chained one after another
JOB1=$(qsub -v JULIA_SCRIPT=$BASE/Budget/SM/perturbation_sm.jl  $PBS)
echo "Job 1 submitted: $JOB1"

JOB2=$(qsub -v JULIA_SCRIPT=$BASE/Budget/SM/FlxDIv_sm.jl  -W depend=afterok:$JOB1 $PBS)
echo "Job 2 submitted: $JOB2"

JOB3=$(qsub -v JULIA_SCRIPT=$BASE/Budget/SM/Conversion_sm.jl  -W depend=afterok:$JOB2 $PBS)
echo "Job 3 submitted: $JOB3"

JOB4=$(qsub -v JULIA_SCRIPT=$BASE/Budget/SM/Ke_b_sm.jl  -W depend=afterok:$JOB3 $PBS)
echo "Job 4 submitted: $JOB4"

JOB5=$(qsub -v JULIA_SCRIPT=$BASE/Budget/SM/Buoyancy_pd_sm.jl  -W depend=afterok:$JOB4 $PBS)
echo "Job 5 submitted: $JOB5"

JOB6=$(qsub -v JULIA_SCRIPT=$BASE/Budget/SM/Shear_pdv_sm.jl -W depend=afterok:$JOB5 $PBS)
echo "Job 6 submitted: $JOB6"

JOB7=$(qsub -v JULIA_SCRIPT=$BASE/Budget/SM/Shear_pdh_sm.jl -W depend=afterok:$JOB6 $PBS)
echo "Job 7 submitted: $JOB7"


# Submit all jobs chained one after another
#JOB1=$(qsub -v JULIA_SCRIPT=$BASE/NIW/UVW_bpfilter.jl  $PBS)
#echo "Job 1 submitted: $JOB1"

JOB8=$(qsub -v JULIA_SCRIPT=$BASE/Budget/SM/Tendency_term.jl -W depend=afterok:$JOB7 $PBS)
echo "Job 8 submitted: $JOB8"

JOB9=$(qsub -v JULIA_SCRIPT=$BASE/Budget/SM/ADV_KE_sm.jl -W depend=afterok:$JOB7 $PBS)
echo "Job 9 submitted: $JOB9"

JOB10=$(qsub -v JULIA_SCRIPT=$BASE/Budget/SM/ADV_PE_sm.jl -W depend=afterok:$JOB9 $PBS)
echo "Job 10 submitted: $JOB10"

JOB11=$(qsub -v JULIA_SCRIPT=$BASE/Budget/SM/Shear_pdh_niw.jl -W depend=afterok:$JOB10 $PBS)
echo "Job 11 submitted: $JOB11"

JOB12=$(qsub -v JULIA_SCRIPT=$BASE/Budget/SM/buoyancy_pd_niw.jl -W depend=afterok:$JOB11 $PBS)
echo "Job 12 submitted: $JOB12"

#JOB13=$(qsub -v JULIA_SCRIPT=$BASE/Budget/NT/Energy_budget_plot_wkly.jl -W depend=afterok:$JOB12 $PBS)
#echo "Job 13 submitted: $JOB13"
# Add as many scripts as you need following the same pattern...




