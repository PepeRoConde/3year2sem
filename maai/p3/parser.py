#!/usr/bin/env python3
import re
import sys

def parse_results(results_text):
    # Split the input text into blocks for each algorithm test
    blocks = re.split(r'Probando\s+', results_text.strip())
    blocks = [b for b in blocks if b]  # Remove empty blocks
    
    parsed_data = []
    
    for block in blocks:
        lines = block.strip().split('\n')
        header = lines[0]
        
        # Extract algorithm name and parameters
        algo_info = {}
        
        # Parse the header
        header_parts = header.split('__')
        algo_info['algorithm'] = header_parts[0]
        
        # Parse parameters
        params = header_parts[1].split('_')
        
        i = 0
        while i < len(params):
            if params[i] == 'de':
                algo_info['de'] = params[i+1]
                i += 2
            elif params[i] == 'pert':
                algo_info['pert'] = params[i+1]
                i += 2
            elif params[i] == 'da':
                algo_info['da'] = params[i+1]
                i += 2
            elif params[i] == 'g':
                # Replace %99 with 0.99
                gamma_value = params[i+1].replace('%', '0.')
                algo_info['g'] = gamma_value
                i += 2
            elif params[i] == 'e':
                # Replace %5 with 0.5
                epsilon_value = params[i+1].replace('%', '0.')
                algo_info['e'] = epsilon_value
                i += 2
            elif params[i] == 'alpha' or params[i] == 'a':
                # Replace %6 with 0.6
                alpha_value = params[i+1].replace('%', '0.')
                algo_info['a'] = alpha_value
                i += 2
            elif params[i] == 'pv':
                algo_info['pv'] = params[i+1]
                i += 2
            else:
                i += 1
        
        # Extract average reward
        for line in lines:
            if line.startswith('Recompensa media'):
                reward = line.split(' ')[-1]
                algo_info['avg_reward'] = reward
                break
        
        parsed_data.append(algo_info)
    
    return parsed_data

def generate_latex_table(parsed_data):
    # Start LaTeX table
    latex = "\\begin{table}[ht]\n"
    latex += "\\centering\n"
    latex += "\\begin{tabular}{|l|c|c|c|c|c|c|c|c|}\n"
    latex += "\\hline\n"
    latex += "Algorithm & $d_e$ & $d_a$ & Pert & $\\gamma$ & $\\alpha$ & $\\epsilon$ & PV & Avg. Reward \\\\ \\hline\n"
    
    # Add data rows
    for data in parsed_data:
        algorithm = data.get('algorithm', '')
        de = data.get('de', '-')
        da = data.get('da', '-')
        pert = data.get('pert', '-')
        gamma = data.get('g', '-')
        alpha = data.get('a', '-')
        epsilon = data.get('e', '-')
        pv = data.get('pv', '-')
        avg_reward = data.get('avg_reward', '-')
        
        # Format the row
        row = f"{algorithm} & {de} & {da} & {pert} & {gamma} & {alpha} & {epsilon} & {pv} & {avg_reward} \\\\ \\hline\n"
        latex += row
    
    # End LaTeX table
    latex += "\\end{tabular}\n"
    latex += "\\caption{Reinforcement Learning Algorithm Performance Comparison}\n"
    latex += "\\label{tab:rl_algorithms}\n"
    latex += "\\end{table}\n"
    
    return latex

def main():
    # Check if a file path was provided
    if len(sys.argv) != 2:
        print("Usage: python parser.py results.txt")
        sys.exit(1)
    
    # Get the file path from command line arguments
    file_path = sys.argv[1]
    
    try:
        # Read the results file
        with open(file_path, 'r', encoding='utf-8') as file:
            results_text = file.read()
        
        # Parse the results
        parsed_data = parse_results(results_text)
        
        # Generate LaTeX table
        latex_table = generate_latex_table(parsed_data)
        
        # Print the LaTeX table
        print(latex_table)
        
        # Optionally save to a .tex file
        output_file = file_path.rsplit('.', 1)[0] + '_table.tex'
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(latex_table)
        
        print(f"LaTeX table saved to {output_file}")
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
