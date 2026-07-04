# import all necessary modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import glob
import ast
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
from tqdm import tqdm
import sys
import os

# function to convert secondary structure to 3 states
def ss_conv(ss):
    secondarystruct = ''
    letter=''
    for char in ss:
        if char in ['H', 'I', 'G']:
            letter = 'H'
        elif char in ['E', 'B']:
            letter = 'E'
        elif char in ['T', 'S']:
            letter = 'L'
        else:
            letter = 'L'
        secondarystruct += letter
    return secondarystruct

# exponential function to model uncertainty
def exponent(x, norm=1.0, tau=0.5):
    if x!=0:
        return norm*np.exp(tau)*np.exp(-1*tau*x)
    else:
        return norm

# step function for the window width based on protein length
def get_window_width(protein_len):
    if protein_len <= 25:
        return 8
    else:
        return 18

# score function for substitution score
def scorefn_substitution(ref_ss,replace_ss):
    """A function to score the substitution of a secondary structure element"""
    assert len(ref_ss) == len(replace_ss), "Secondary structures are not the same length"
    score = 0
    for i in range(len(ref_ss)):
        if ref_ss[i] == 'H' or ref_ss[i] == 'I' or ref_ss[i] == 'G':
            if replace_ss[i] == 'E' or replace_ss[i] == 'B':
                score += 0.7
            elif replace_ss[i] == 'T':
                score += 0.3
        elif ref_ss[i] == 'E' or ref_ss[i] == 'B':
            if replace_ss[i] == 'H' or replace_ss[i] == 'I' or replace_ss[i] == 'G':
                score += 0.7
            elif replace_ss[i] == 'T':
                score += 1
        elif ref_ss[i] == 'T':
            if replace_ss[i] == 'H' or replace_ss[i] == 'I' or replace_ss[i] == 'G':
                score += 0.3
            elif replace_ss[i] == 'E' or replace_ss[i] == 'B':
                score += 1
    return score/len(ref_ss)

# score function for diversity Chen et al. 2018
def scorefn_chen(h,e,l,offset=0):
    return 1/(h**2+e**2+l**2)

# score function for information entropy
def scorefn_entropy(h,e,l):
    #check if any are zero
    if h == 0:
        h = 0.00001
    if e == 0:
        e = 0.00001
    if l == 0:
        l = 0.00001
    return -1*(h*np.log(h)+e*np.log(e)+l*np.log(l))


# metrics function definition
def metrics(data):
    """A function to calculate the metrics regarding the results of the motif search"""    
    
    #parse row by row through the data
    frag_data = {}      # dict to store the details pertaining to one fragment window
    sec_struct = []
    plddt = []
    cluster_indices = []
    for row in data:
        if row != []:
            if not row[0].isdigit() and row[1] != "0 hits":
                sec_struct.append(row[3])
                plddt.append(ast.literal_eval(row[4]))
                assert len(sec_struct) == len(plddt), "Error: The length of the secondary structure and pLDDT lists are not the same"
                cluster_indices.append(row[6])
            else:
                if row[0].isdigit():
                    frag_data[row[0]+row[1]] = [row[2],sec_struct,plddt,cluster_indices]
                if row[1] == "0 hits":
                    frag_data[row[0]+row[1]] = [0,[],[],[],]
                sec_struct = []
                plddt = []
                cluster_indices = []

    # Filter the fragments with pLDDT > 70 and secondary structure content
    frag_data_filtered = {}
    for key, value in frag_data.items():
        sec_struct = value[1]
        plddt = value[2]
        cluster_index = value[3]

        # Create new lists for filtered values
        filt_sec_struct = []
        filt_plddt = []
        filt_cluster_indices = []

        # First pass: filter out items where sec_struct has 'U' at middle_rsd_index
        for i in range(len(sec_struct)):
            if sec_struct[i][middle_rsd_index] != 'U':
                filt_sec_struct.append(sec_struct[i])
                filt_plddt.append(plddt[i])
                filt_cluster_indices.append(cluster_index[i])
        
        # Second pass: filter out items where pLDDT < plddt_cutoff
        final_sec_struct = []
        final_plddt = []
        final_cluster_indices = []

        for i in range(len(filt_plddt)):
            if len(filt_plddt[i]) == 0:
                final_sec_struct.append(filt_sec_struct[i])
                final_plddt.append(filt_plddt[i])
                final_cluster_indices.append(filt_cluster_indices[i])
                continue
            if float(filt_plddt[i][middle_rsd_index]) >= plddt_cutoff:
                final_sec_struct.append(filt_sec_struct[i])
                final_plddt.append(filt_plddt[i])
                final_cluster_indices.append(filt_cluster_indices[i])

        # replacing poly-proline helix with loop secondary structure
        for i in range(len(final_sec_struct)):
            if 'P' in final_sec_struct[i]:
                final_sec_struct[i].replace("P","L")

        assert len(final_sec_struct) == len(final_plddt), "The length of the filtered secondary structure and pLDDT lists are not the same"
        frag_data_filtered[key] = [len(final_sec_struct), final_sec_struct, final_plddt, final_cluster_indices]

    # calculate the metrics
    scores_dict = {}
    for key, value in frag_data_filtered.items():
        if len(value[1]) != 0:
            # 1. Substitution score    
            #sort the secondary structures based on the number of occurences
            most_comm_ss = Counter(value[1]).most_common(1)[0]
            substitution_scores = []
            for ss in value[1]:
                substitution_scores.append(scorefn_substitution(most_comm_ss[0],ss))

            if most_comm_ss[1] == len(value[1]):
                scores_dict[key]=[0]
            else:
                final_substitution_score = 0
                for i in range(len(substitution_scores)):
                    final_substitution_score += substitution_scores[i]*(1/value[3].count(value[3][i]))
                num_nonzero = len([x for x in substitution_scores if x != 0])
                if not num_nonzero == 0:
                    final_substitution_score = final_substitution_score/num_nonzero
                else:
                    final_substitution_score = 0
                scores_dict[key]=[final_substitution_score]

            #2. Diversity and entropy scores
            middle_ss = [ss_conv(x[middle_rsd_index]) for x in value[1]]
            ss_helix = 0
            ss_sheet = 0
            ss_coil = 0
            for i in range(len(middle_ss)):
                if middle_ss[i] == 'H':
                    ss_helix += 1*(1/value[3].count(value[3][i]))
                elif middle_ss[i] == 'E':
                    ss_sheet += 1*(1/value[3].count(value[3][i]))
                elif middle_ss[i] == 'L':
                    ss_coil += 1*(1/value[3].count(value[3][i]))
            total = ss_helix + ss_sheet + ss_coil
            diversity = scorefn_chen(ss_helix/total,ss_sheet/total,ss_coil/total)
            entropy = scorefn_entropy(ss_helix/total,ss_sheet/total,ss_coil/total)

            scores_dict[key].append(diversity)
            scores_dict[key].append(entropy)

            #3. Uncertainty score
            scores_dict[key].append(exponent(len(set(value[3])),norm=1.0,tau=0.5))

            #4. Helix, sheet and coil probabilities
            middle_ss = [ss_conv(x[middle_rsd_index]) for x in value[1]]
            ss_helix = 0
            ss_sheet = 0
            ss_coil = 0
            for i in range(len(middle_ss)):
                if middle_ss[i] == 'H':
                    ss_helix += 1*(1/value[3].count(value[3][i]))
                elif middle_ss[i] == 'E':
                    ss_sheet += 1*(1/value[3].count(value[3][i]))
                elif middle_ss[i] == 'L':
                    ss_coil += 1*(1/value[3].count(value[3][i]))
            total = ss_helix + ss_sheet + ss_coil
            scores_dict[key].append(ss_helix/total)
            scores_dict[key].append(ss_sheet/total)
            scores_dict[key].append(ss_coil/total)
        else:
            # No hits found
            scores_dict[key]=[0]
            scores_dict[key].append(1)
            scores_dict[key].append(0)
            scores_dict[key].append(1)
            scores_dict[key].append(0)
            scores_dict[key].append(0)
            scores_dict[key].append(0)

    df = pd.DataFrame.from_dict(scores_dict,orient='index',columns=['substitution_score','diversity','entropy','uncertainty','helix_prob','sheet_prob','coil_prob'])
    return df

def task(file,final_scores_dict,outpath,plot_results=False):
    '''Task function for multiprocessing'''
    # read the results tsv file
    data = csv.reader(open(file), delimiter="\t")

    # calculate all the metrics
    file_df = metrics(data)
    length_of_protein = len(file_df) + 6
    window_width = get_window_width(length_of_protein)
    substitution_rolling_avg = file_df['substitution_score'].rolling(window=window_width, center=True).mean()
    entr_rolling_avg = file_df['entropy'].rolling(window=window_width,center=True).mean()
    div_rolling_avg = file_df['diversity'].rolling(window=window_width,center=True).mean()
    uncert_score = file_df['uncertainty'].mean()
    uncert_rolling_avg = file_df['uncertainty'].rolling(window=window_width,center=True).mean() # uncertainty rolling average for plotting

    if plot_results:
        # plot the metrics
        # Set seaborn style for prettier plots
        protein_len = len(file_df)
        sns.set_theme(style="whitegrid")
        # Get screen size
        # Create a figure with screen size
        fig, axes = plt.subplots(2, 2, figsize=(20,15))
        fig.suptitle(os.path.basename(file)[:-4] + ' Diversity Metrics',fontsize=15)

        # First subplot: Helix and sheet probability (line plot)
        sns.lineplot(x=range(protein_len), y=file_df['helix_prob'], ax=axes[0, 0], label='Helix Probability')
        sns.lineplot(x=range(protein_len), y=file_df['sheet_prob'], ax=axes[0, 0], label='Sheet Probability')
        # minimum of helix and sheet probability
        min_prob = [min(list(file_df['helix_prob'])[i], list(file_df['sheet_prob'])[i]) for i in range(len(list(file_df['helix_prob'])))]
        axes[0, 0].fill_between(range(protein_len),np.zeros(protein_len),min_prob,color='grey',alpha=0.5, label='Minimum of the two')
        axes[0, 0].set_title('Helix and Sheet Probability', fontsize=15)
        axes[0, 0].set_xlabel('Residue', fontsize=13)
        axes[0, 0].set_ylabel('Probability', fontsize=13)
        axes[0, 0].set_ylim(0,1)
        axes[0, 0].legend(loc='upper left', fontsize=11)

        # Second subplot: substitution score (line plot)
        sns.lineplot(x=range(protein_len), y=file_df['substitution_score'], ax=axes[0, 1], label='substitution')
        # rolling average of the substitution score
        sns.lineplot(x=range(protein_len), y=substitution_rolling_avg, ax=axes[0, 1], label='Substitution Rolling Average')
        axes[0, 1].set_title('Substitution Score', fontsize=15)
        axes[0, 1].set_xlabel('Residue', fontsize=13)
        axes[0, 1].set_ylabel('Substitution Score', fontsize=13)
        axes[0, 1].set_ylim(0,1)
        axes[0, 1].legend(loc='upper left', fontsize=11)

        # Third subplot: Entropy score (line plot)
        sns.lineplot(x=range(protein_len), y=file_df['entropy'], ax=axes[1, 0], label='Entropy')
        # rolling average of the entropy score
        sns.lineplot(x=range(protein_len), y=entr_rolling_avg, ax=axes[1, 0], label='Entropy Rolling Average')
        axes[1, 0].set_title('Entropy Score', fontsize=15)
        axes[1, 0].set_xlabel('Residue', fontsize=13)
        axes[1, 0].set_ylabel('Entropy Score', fontsize=13)
        axes[1, 0].set_ylim(0,1)
        axes[1, 0].legend(loc='upper left', fontsize=11)

        # Fourth subplot: Diversity score (line plot)
        sns.lineplot(x=range(protein_len), y=file_df['diversity'], ax=axes[1, 1], label='Diversity')
        # rolling average of the diversity score
        sns.lineplot(x=range(protein_len), y=div_rolling_avg, ax=axes[1, 1], label='Diversity Rolling Average')
        # plot uncertainty + diversity and uncertainty - diversity as a shaded region
        axes[1, 1].fill_between(range(protein_len),uncert_rolling_avg+div_rolling_avg,div_rolling_avg-uncert_rolling_avg,color='grey',alpha=0.5, label='Uncertainty')
        axes[1, 1].set_title('Diversity Score', fontsize=15)
        axes[1, 1].set_xlabel('Residue', fontsize=13)
        axes[1, 1].set_ylabel('Diversity Score', fontsize=13)
        axes[1, 1].set_ylim(1,3)
        axes[1, 1].legend(loc='upper left', fontsize=11)

        # Adjust layout
        plt.tight_layout()

        # Increase the space between subplots
        plt.subplots_adjust(hspace=0.4, wspace=0.3)

        # Save the figure
        fig.savefig(outpath + os.path.basename(file)[:-4] + '.png',bbox_inches='tight')
        plt.close(fig)

    final_scores_dict[os.path.basename(file)[:-4]] = [
        div_rolling_avg.dropna().max(),
        entr_rolling_avg.dropna().max(),
        substitution_rolling_avg.max(),
        uncert_score,
    ]

def task_wrapper(file, final_scores_dict, output_path,plot_bool, queue):
    try:
        task(file, final_scores_dict, output_path, plot_results=plot_bool)  # Make sure 'task' is accessible
    except Exception as e:
        print(f"Error processing {file}: {e}")
    finally:
        queue.put(1)

def main():
    # get input path from command line argument
    prot_type = sys.argv[1]
    results_path = sys.argv[2]
    output_path = sys.argv[3]
    plot_diversity = sys.argv[4] == "True"

    # use glob to get all the tsv files in the input path
    infiles = glob.glob(results_path + "*.tsv")

    # initialize multiprocessing for parallel processing
    mp_manager = mp.Manager()
    final_scores_dict = mp_manager.dict()
    progress_queue = mp_manager.Queue()
    pool = mp.Pool(mp.cpu_count())

    # map the task to include the queue
    pool.starmap_async(task_wrapper, [(file, final_scores_dict, output_path,plot_diversity, progress_queue) for file in infiles])

    # initialize tqdm progress bar
    pbar = tqdm(total=len(infiles), desc="Calculating Diversity")

    # update progress bar based on queue
    for _ in range(len(infiles)):
        progress_queue.get()
        pbar.update(1)

    pbar.close()
    pool.close()
    pool.join()

    # convert manager dict to regular dict for easier handling
    final_scores_dict = dict(final_scores_dict)
    final_csv = output_path + prot_type + '.csv'

    with open(final_csv, 'w') as f:
        #write header to the file
        f.write('PDB_ID\tDiversity Score\tEntropy Score\tSubstitution Score\tUncertainty\tPrediction\tConfidence\n')
        #create a dataframe to store the scores
        final_scores_df = pd.DataFrame.from_dict(final_scores_dict,orient='index',columns=['Diversity Score','Entropy Score','Substitution Score','Uncertainty']) 
        #sort the dataframe based on the diversity score
        final_scores_df = final_scores_df.sort_values(by='Diversity Score',ascending=False)
        #write the scores to a csv file, first round the scores to 5 decimal places
        final_scores_df = final_scores_df.round(5)
        #change nan values to 'NA' if any
        final_scores_df = final_scores_df.fillna('NA')
        final_scores_df.to_csv(f, sep='\t', header=False)
    print("Metrics calculated and saved to: ", final_csv)

if __name__ == "__main__":
    frag_size = 7   # 7-mer fragments
    middle_rsd_index = 2   # 3rd residue in the fragment
    plddt_cutoff = 70   # pLDDT cutoff for filtering
    main()