J Kyle Medley
## libRoadRunner/Jarnac Benchmarking

## Usage

You must have a deployment of libRoadRunner version 1.3.1 in your Python module 
search path before continuing. The benchmarking scripts can be run as follows:
```
git clone -b submission https://github.com/0u812/rr-benchmarking.git
cd rr-benchmarking
```

## libRoadRunner

To run the roadrunner ODE benchmark and save the results to rr-ode-results.csv,
execute the following command from the Windows command prompt:
```
python ./rr_bench_ode.py >rr-ode-results.csv
```

To run the roadrunner stochastic benchmark and save the results to rr-ode-results.csv,
execute the following command from the Windows command prompt:
```
python ./rr_bench_stoch.py >rr-stoch-results.csv
```

## Jarnac

To run the Jarnac ODE benchmark, open the file `jarnac_bench_ode.jan` in the Jarnac
IDE and click the "Run" button. The results will be displayed in the interactive console.
Copy and paste the output to a csv file. This benchmark does not include the Jean-Marie
AMPA receptor due to the fact that Jarnac cannot import the SBML directly. For this model,
a separate benchmark is included.

To run the Jean-Marie AMPA receptor ODE benchmark, open `jean_marie_converted_bench.jan`
in the Jarnac IDE and click the "Run" button. The results will be displayed in the interactive console.
Copy and paste the output to a csv file.

To run the Jarnac stochastic/Gillespie benchmark, open the file `jarnac_bench_stoch.jan` in the Jarnac
IDE and click the "Run" button. Copy and paste the output to a csv file.