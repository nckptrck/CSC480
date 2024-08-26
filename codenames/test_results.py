import subprocess
import csv
import json
import os

codemasters = ['players.codemaster_w2v_wn.AICodemaster',
                'players.codemaster_w2v_03.AICodemaster',
                'players.codemaster_w2v_05.AICodemaster',
                'players.codemaster_w2v_07.AICodemaster',
                'players.codemaster_wn_lin.AICodemaster']

guessers = ['players.guesser_w2v.AIGuesser', 
            'players.guesser_wn_lin.AIGuesser']

codemaster_mapping = {
    'players.codemaster_w2v_wn.AICodemaster': 'W2V and WordNet',
    'players.codemaster_w2v_03.AICodemaster': 'W2V 03',
    'players.codemaster_w2v_05.AICodemaster': 'W2V 05',
    'players.codemaster_w2v_07.AICodemaster': 'W2V 07',
    'players.codemaster_wn_lin.AICodemaster': 'WordNet Lin'
}

guesser_mapping = {
    'players.guesser_w2v.AIGuesser': 'W2V',
    'players.guesser_wn_lin.AIGuesser': 'WordNet Lin'
}

def run_test_games():
    output_csv = "results/codenames_results.csv"
    fieldnames = ['game_name', 'total_turns', 'R', 'B', 'C', 'A', 'codemaster', 'guesser', 'seed', 'time_s']

    for codemaster in codemasters:
        print("codemaster: ", codemaster)
        for guesser in guessers:
            print("guesser: ", guesser)
            seed = 100
            for i in range(30):
                subprocess.run(["python", "run_game.py", codemaster, guesser, "--w2v",
                        "players/GoogleNews-vectors-negative300.bin", "--wordnet", "ic-brown.dat",
                        "--seed", str(seed)])
                seed += 50

                if os.path.exists("results/bot_results_new_style.txt"):
                    last_line = read_last_line("results/bot_results_new_style.txt")
                    result = json.loads(last_line)
                    result_row = {field: result.get(field) for field in fieldnames}

                    codemaster_full = codemaster_mapping.get(codemaster, result_row['codemaster'])
                    guesser_full = guesser_mapping.get(guesser, result_row['guesser'])
                    result_row['codemaster'] = codemaster_full
                    result_row['guesser'] = guesser_full

                    with open(output_csv, mode='a', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=fieldnames)
                        writer.writerow(result_row)


def read_last_line(file_path):
    """Read the last line of a file by reading the entire file."""
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            return lines[-1] if lines else None
    except Exception as e:
        print(f"Error reading last line: {e}")
        return None
    

if __name__ == "__main__":
    run_test_games()


