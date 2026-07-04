# Morpheus: 1.0
# Given a proteome to perform fragment picking on, Morpheus stores the query 
# fragments from the proteome in the form of a trie and parses through the database
# to return all matches to the original trie's leaf nodes and saves the final results in a tsv file.

import pygtrie
import ast
from tqdm import tqdm
import pickle
import os.path
import subprocess
import multiprocessing as mp
from glob import glob
import sys

fragpicker_dir = os.path.dirname(os.path.realpath(__file__))    # path to the fragpicker directory
fragpicker_dir = os.path.join(fragpicker_dir, "")       # add a trailing slash based on OS
parent_dir = os.path.abspath(os.path.join(fragpicker_dir, os.pardir))   # path to the parent directory
parent_dir = os.path.join(parent_dir, "")   # add a trailing slash based on OS

reference_file = os.path.join(parent_dir,"database","fragment_search_db.fasta")     # database file for fragment search
infile_path = os.path.join(fragpicker_dir,"input_proteomes","")      # path to the input proteomes - add the proteome fasta file here, check README for more details
pickle_location = os.path.join(fragpicker_dir,"output","pickle_files","")   # path to store the pickle files, use the output/pickle_files folder in the code directory
outfile_location = os.path.join(fragpicker_dir,"output","")   # path to store the tsv files, use the output folder in the code directory
proteomes = glob(infile_path+"*.fasta")    # get all the proteomes in the input_proteomes folder
plot_diversity = False     # set to True if you want to plot the diversity metrics, recommended only in the case of few proteins


for proteome in proteomes:
    proteome_name = os.path.basename(proteome).split(".")[0]
    print("Working on proteome", proteome_name)
    fasta_file = proteome
    proteome_outfile_location = os.path.join(outfile_location, proteome_name, "")

    # make the output directory if it doesn't exist
    if not os.path.exists(proteome_outfile_location):
        os.makedirs(proteome_outfile_location)

    if not os.path.exists(pickle_location+proteome_name+'.pkl'):
        # read fasta file and make a trie with all the 7-mer fragments
        print("Reading fasta file and making a trie with all the 7-mer fragments")
        nmer_trie = pygtrie.StringTrie()
        with open(fasta_file, 'r') as f:
            for line in tqdm(f, desc="Reading fasta file"):
                if line.startswith(">"):
                    continue
                else:
                    line = line.strip()
                    for i in range(0, len(line) - 6):
                        fragment = line[i:i + 7]
                        if fragment in nmer_trie:
                            nmer_trie[fragment] += []
                        else:
                            nmer_trie[fragment] = []
        
        # search the database for the fragment and return the matches to the original trie leaf nodes
        print("Reading reference file and searching for the fragments")
        with open(reference_file, 'r') as f:
            for line in tqdm(f, total=3643900, desc="Reading reference file"):
                if line.startswith(">"):
                    seq_info = line.split(":")
                    seq_info[-1] = ast.literal_eval(seq_info[-1].strip())
                else:
                    aa_seq = line.strip()
                    seq_len = len(aa_seq)
                    seq_info_last = seq_info[-1]
                    seq_info_second_last = seq_info[-2]
                    cluster_id = seq_info[1]
                    for i in range(seq_len - 6):
                        fragment = aa_seq[i:i + 7]
                        if fragment in nmer_trie:
                            nmer_trie[fragment].append([
                                seq_info[0][1:], 
                                i + 1, 
                                i + 7, 
                                seq_info_second_last[i:i + 7], 
                                seq_info_last[i:i + 7], 
                                aa_seq,
                                cluster_id
                            ])

        #save the final trie to a pickle_file file
        print("Saving the trie to a file")
        with open(pickle_location+proteome_name+'.pkl', 'wb') as f:
            pickle.dump(nmer_trie, f)
    else:
        with open(pickle_location+proteome_name+'.pkl', 'rb') as f:
            print("Loading the trie from the file")
            nmer_trie = pickle.load(f)

    # make a tsv file corresponding to all proteins in the proteome
    #find number of lines in the fasta file
    with open(fasta_file, 'r') as f:
        fasta_nlines = sum(1 for _ in f)
    
    # Open the FASTA file
    with open(fasta_file, 'r') as f:
        uniprot_id = None
        file_handle = None

        for line in tqdm(f, total=int(fasta_nlines), desc="Making tsv files"):
            if line.startswith(">"):
                if file_handle:
                    file_handle.close()

                uniprot_id = line.split("|")[1]
                file_path = os.path.join(proteome_outfile_location, uniprot_id + ".tsv")
                file_handle = open(file_path, 'w')
            
            else:
                if file_handle:
                    content = []
                    line_strip = line.strip()
                    line_length = len(line_strip)

                    for i in range(0, line_length - 6):
                        fragment = line_strip[i:i + 7]
                        matches = nmer_trie.get(fragment, [])
                        if len(matches) > 10000:
                            matches = matches[:10000]

                        if matches:
                            for match in matches:
                                content.append(f"{match[0]}\t{match[1]}\t{match[2]}\t{match[3]}\t[{str(match[4])[1:-1]}]\t{match[5]}\t{match[6]}\n")
                            content.append(f"{i + 1}\t{fragment}\t{len(matches)}\n\n")
                        else:
                            content.append(f"{i + 1}\t{fragment}\t0\n\n")
                    
                    file_handle.writelines(content)
        
        if file_handle:
            file_handle.close()


    # call calculate_diversity_metrics.py using subprocess
    print("Calculating diversity metrics")
    subprocess.run([sys.executable, fragpicker_dir+"calculate_diversity_metrics.py", proteome_name, proteome_outfile_location, outfile_location, str(plot_diversity)])

    # # uncomment if necessary (linux only)
    # # remove the tsv files once the diversity metrics are calculated
    # print("Removing tsv files")
    # subprocess.run(["rm", "-r", outfile_location])
    
    # done!
    print("done with proteome", proteome_name)
    print("---------------------------------")
