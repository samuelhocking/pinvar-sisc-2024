# Execute a cross-validation run

path=$(pwd)
pinvar_path="${path%/pinvar*}/pinvar"

# change as necessary to directory containing the modules:
# Pinvar.jl
# Odes.jl
module_path=$pinvar_path/modules
scripts_path=${pinvar_path}/scripts
data_path=${pinvar_path}/data

# add results folder if necessary
results_path=${path}/results
	if [[ ! -d $results_path ]]
	then
		mkdir $results_path
	fi

echo "data_path    : $data_path"
echo "results_path : $results_path"

ode_name=lv
h=h2

julia ${scripts_path}/run_results_${h}.jl $ode_name $module_path $data_path $results_path 