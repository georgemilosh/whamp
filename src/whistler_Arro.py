import subprocess
import os
import sys
import re
import pandas as pd
import numpy as np

def get_ion_electron_mass_ratio(filepath):
    """
    Extract only the Ion to electron mass ratio from WHAMP model file.
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    # Get non-comment lines
    data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    
    # The Ion to electron mass ratio is the last (10th) data line
    if len(data_lines) >= 10:
        return float(data_lines[9])
    else:
        raise ValueError("File doesn't contain enough data lines")

# Cell 2: Read the WHAMP output file with proper parsing
def read_whamp_output_robust(filename):
    """
    Most robust WHAMP output parser using tokenization approach.
    
    Parameters:
    filename: path to the text file containing WHAMP output
    
    Returns:
    DataFrame with columns extracted from the WHAMP output format
    """
    import re
    
    def tokenize_line(line):
        """Tokenize a line into variable=value pairs"""
        # Pattern to match scientific notation numbers
        number_pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
        
        # Pattern to match variable=value sequences
        var_pattern = rf'(\w+)=\s*({number_pattern}(?:\s+{number_pattern})*)'
        
        matches = re.findall(var_pattern, line)
        
        parsed = {}
        for var_name, value_string in matches:
            # Extract all numbers from the value string
            numbers = re.findall(number_pattern, value_string)
            
            # Keep important variables in uppercase for consistency
            important_vars = {'A', 'BETA', 'P', 'Z', 'F'}
            if var_name.upper() in important_vars:
                var_name_key = var_name.upper()
            else:
                var_name_key = var_name.lower()
            
            try:
                if len(numbers) == 1:
                    # Single value
                    val = float(numbers[0])
                    if var_name.lower() == 'f':
                        parsed['omega_r'] = val
                    else:
                        parsed[var_name_key] = val
                        
                elif len(numbers) == 2:
                    # Complex pair
                    real_val, imag_val = float(numbers[0]), float(numbers[1])
                    if var_name.lower() == 'f':
                        parsed['omega_r'] = real_val
                        parsed['omega_i'] = imag_val
                    else:
                        parsed[f'{var_name_key}_real'] = real_val
                        parsed[f'{var_name_key}_imag'] = imag_val
                        
                else:
                    # Multiple values
                    for i, num in enumerate(numbers):
                        parsed[f'{var_name_key}_{i+1}'] = float(num)
                        
            except ValueError:
                # If conversion fails, store as strings
                parsed[var_name_key] = value_string
        
        return parsed
    
    try:
        data = []
        
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                
                # Skip comment lines
                if line.startswith('#'):
                    continue
                
                # Parse the line
                row = tokenize_line(line)
                
                # Only add if we found meaningful data
                if any(key in row for key in ['P', 'Z', 'omega_r', 'f']):
                    data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        return df
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
def read_whamp_output(filename):
    """
    Read WHAMP output data from text file into a pandas DataFrame.
    
    Parameters:
    filename: path to the text file containing WHAMP output
    
    Returns:
    DataFrame with columns extracted from the WHAMP output format
    """
    try:
        data = []
        
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                # Initialize row data
                row = {}
                
                # Extract p value - more specific pattern
                p_match = re.search(r'p=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
                if p_match:
                    row['p'] = float(p_match.group(1))
                
                # Extract z value - more specific pattern
                z_match = re.search(r'z=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
                if z_match:
                    row['z'] = float(z_match.group(1))
                
                # Extract f (frequency) - handle scientific notation properly
                # Pattern: f= number number (where numbers can be in scientific notation)
                f_match = re.search(r'f=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
                if f_match:
                    row['omega_r'] = float(f_match.group(1))
                    row['omega_i'] = float(f_match.group(2))
                
                # Extract EX (electric field X component) - handle spacing properly
                ex_match = re.search(r'EX=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
                if ex_match:
                    row['EX_real'] = float(ex_match.group(1))
                    row['EX_imag'] = float(ex_match.group(2))
                
                # Extract EY (electric field Y component)
                ey_match = re.search(r'EY=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
                if ey_match:
                    row['EY_real'] = float(ey_match.group(1))
                    row['EY_imag'] = float(ey_match.group(2))
                
                # Extract EZ (electric field Z component)
                ez_match = re.search(r'EZ=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
                if ez_match:
                    row['EZ_real'] = float(ez_match.group(1))
                    row['EZ_imag'] = float(ez_match.group(2))
                
                # Extract BETA - handle scientific notation
                beta_match = re.search(r'BETA=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
                if beta_match:
                    row['BETA'] = float(beta_match.group(1))
                
                # Extract A (alpha parameter) - more specific pattern to avoid matching BETA
                a_match = re.search(r'A=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$', line)
                if a_match:
                    row['A'] = float(a_match.group(1))
                
                # Only add row if we found at least p, z, and frequency data
                if 'p' in row and 'z' in row and 'omega_r' in row:
                    data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        return df
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def run_whamp_automated(model_file, output_file, max_iterations=1900, commands=None):
    """
    Run WHAMP with automated input commands
    
    Parameters:
    - model_file: path to the model file (e.g., '../Models/H17f3c')
    - output_file: path to the output file (e.g., '../results/parallel_firehose5.txt')
    - max_iterations: maximum number of iterations (default: 1900)
    - commands: list of commands to send to WHAMP (default: standard firehose commands)
    """
    
    if commands is None:
        # Default commands for parallel firehose instability analysis
        commands = [
            'p0z.1,1,-.1f1e-4',  # Set P and Z ranges with start frequency
            'pzfewa',            # Set output format
            'S'                  # Stop/quit
        ]
    
    # Build the command
    whamp_cmd = [
        './whamp',
        '-maxiterations', str(max_iterations),
        '-file', model_file,
        '-outfile', output_file
    ]
    
    print(f"Running WHAMP with command: {' '.join(whamp_cmd)}")
    #print(f"Input commands: {commands}")
    
    try:
        # Start the WHAMP process
        process = subprocess.Popen(
            whamp_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Send commands to WHAMP
        for i, command in enumerate(commands):
            print(f"Sending command {i+1}: {command}")
            process.stdin.write(command + '\n')
            process.stdin.flush()
            
            # Small delay to allow WHAMP to process the command
            #time.sleep(0.1)
        
        # Wait for the process to complete
        stdout, stderr = process.communicate(timeout=60)  # 60 second timeout
        
        print("WHAMP output:")
        print(stdout)
        
        if stderr:
            print("WHAMP errors:")
            print(stderr)
        
        if process.returncode == 0:
            print(f"WHAMP completed successfully!")
            print(f"Output saved to: {output_file}")
            return True
        else:
            print(f"WHAMP failed with return code: {process.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("WHAMP process timed out!")
        process.kill()
        return False
    except Exception as e:
        print(f"Error running WHAMP: {e}")
        return False

def run_whamp_parameter_sweep(model_file, output_base, a_values, max_iterations=1900):
    """
    Run WHAMP for multiple temperature anisotropy values
    
    Parameters:
    - model_file: path to the model file
    - output_base: base name for output files (will append _A_value.txt)
    - a_values: list of A values to test (e.g., [0.1, 0.2, 0.3, 0.4, 0.5])
    - max_iterations: maximum number of iterations
    """
    
    results = []
    
    for a_val in a_values:
        output_file = f"{output_base}_A_{a_val:.3f}.txt"
        
        commands = [
            'p0z.1,1,-.1f1e-4',  # Set P and Z ranges with start frequency
            'pzfewa',            # Set output format
            f'a{a_val}',         # Set temperature anisotropy
            'p0z.1,1,-.1f1e-4',  # Re-run with new A value
            'S'                  # Stop/quit
        ]
        
        print(f"\n{'='*60}")
        print(f"Running WHAMP for A = {a_val}")
        print(f"{'='*60}")
        
        success = run_whamp_automated(model_file, output_file, max_iterations, commands)
        results.append({
            'A_value': a_val,
            'output_file': output_file,
            'success': success
        })
    
    return results

def run_whamp_interactive_realtime(model_file, output_file, max_iterations=1900):
    """
    Run WHAMP with real-time interactive input (shows WHAMP output as it runs)
    """
    
    whamp_cmd = [
        './whamp',
        '-maxiterations', str(max_iterations),
        '-file', model_file,
        '-outfile', output_file
    ]
    
    print(f"Running WHAMP interactively: {' '.join(whamp_cmd)}")
    
    try:
        process = subprocess.Popen(
            whamp_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Commands to send
        commands = [
            'p0z.1,1,-.1f1e-4',
            'pzfewa',
            'S'
        ]
        
        command_index = 0
        
        # Real-time interaction
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            
            if output:
                print(output.strip())
                
                # Check if WHAMP is asking for input
                if '#INPUT:' in output:
                    if command_index < len(commands):
                        command = commands[command_index]
                        print(f"Sending: {command}")
                        process.stdin.write(command + '\n')
                        process.stdin.flush()
                        command_index += 1
                    else:
                        print("No more commands to send")
                        break
        
        # Wait for completion
        process.wait()
        
        return process.returncode == 0
        
    except Exception as e:
        print(f"Error in interactive mode: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Make sure we're in the right directory
    if not os.path.exists('./whamp'):
        print("Error: whamp executable not found in current directory")
        print("Please run this script from the whamp build directory")
        sys.exit(1)

    a_values = np.logspace(np.log10(1.0), np.log10(10.0), num=20)  # logarithmically spaced between 0.1 and 1.0 inclusive
    #c_values = np.logspace(np.log10(2*0.09269), np.log10(0.09269/16), num=20)  # GJGR85 model
    c_values = np.logspace(np.log10(2*0.317423), np.log10(0.317423/64), num=20)  # peppe model

    print(f"Running parameter sweep for A values: {a_values}")
    print(f"Running parameter sweep for C values: {c_values}")
    commands = ['p0z25f1e8', 'pzfebwa']#commands = ['p0z25f1e8', 'pzfebwa(2)']  #['p0z25f1e6', 'pzfebwa(2)']
    #     Cell 3: Load your data
    #filename = '/Users/u0167590/github/whamp/results/parallel_whistler.txt'
    #model = '../Models/GJGR85' # normal whistler model

    filename = '/Users/u0167590/github/whamp/results/whistler_Arro.txt'
    model = '../Models/Arro' # Model from F(1) Arrò, G.; Califano, F.; Lapenta, G. Spectral Properties and Energy Transfer at Kinetic Scales in Collisionless Plasma Turbulence. A&A 2022, 668, A33. https://doi.org/10.1051/0004-6361/202243352.
    mass_ratio = get_ion_electron_mass_ratio(model)

    for c in c_values:
        for a in a_values:
            z_max = 10
            if c > 0.25515:
                f = .1
                z_min = .1
            else:
                f = .01
                z_min = .3
                if a < 1.44:
                    f = .001
                if c < 0.022642:
                    z_max = 50
            
        
            dz = z_min / 20
            

            #if c >= 0.21264:
            #    z_min = .3
            #    dz = z_min / 20
            #    f = .5
            
            #if c <= 0.04321693533:
            #    z_max = 800./np.sqrt(mass_ratio) # z_max = 40
            #    if a < 1.44:
            #        z_min = 30/np.sqrt(mass_ratio)    # z_min = 3
            #        f = 1/mass_ratio # f = .01

            commands.append(f'c{c}a{a}f{f}') #commands.append(f'c{c}a(2){a}f{f}')
            commands.append(f'z{z_min},{z_max},{dz}')
            commands.append(f'z0,{z_min-dz},-{dz}')
    commands.append('S')
    print("Example 1: Single WHAMP run")
    success = run_whamp_automated(
        model_file=model,
        output_file=filename, max_iterations=1200,
        commands=commands
    )
    
    if success:
        print("\Parameter sweep runs completed successfully!")
    
    
    df = read_whamp_output_robust(filename)
    # Display basic info about the data
    if df is not None:
        print(f"Data loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"\nColumn names: {list(df.columns)}")
        print(f"\nUnique A values: {sorted(df['A'].unique())}")
        print(f"\nUnique BETA values: {sorted(df['BETA'].unique())}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        # Group by A value to see data structure
        print(f"\nData grouped by A value:")
        for beta_val in sorted(df['BETA'].unique()):
            print(f"\nBETA = {beta_val}:")
            for a_val in sorted(df['A'].unique()):
                subset = df[(df['A'] == a_val) & (df['BETA'] == beta_val)]
                subset1e3 = df[(df['A'] == a_val) & (df['BETA'] == beta_val) & (df['omega_r'] > 1e5)]
                print(f"A = {a_val}, BETA = {beta_val}: {len(subset)} entries with {len(subset1e3)} entries above omega_r > 1e5")
        print(f" Total number of entries {len(df)}, total number of omega_r > 1e5: {len(df[df['omega_r'] > 1e5])}")
    else:
        print("Failed to load data")