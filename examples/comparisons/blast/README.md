# BLAST benchmark comparison

To run the BLAST benchmark you need to have BLAST installed by the following command:
```bash
conda install bioconda::blast
```

First, download the Pfam20 dataset, by running:
```bash
python -m protein_search_evals.datasets.pfam
```

Then, set up the BLAST input files:
```bash
mkdir blast # create a directory for the BLAST database
cd blast # change to the blast directory
cp ../data/pfam/pfam20_seed-42/sequences.fasta pfam20_seed-42.fasta
cp pfam20_seed-42.fasta query.fasta
```

Then, make the BLAST database using the following commands:
```bash
makeblastdb -in pfam20_seed-42.fasta -dbtype prot -out pfam_db
```

Then, run the BLAST benchmark using the following command:
```bash
nohup blastp -query query.fasta -db pfam_db -outfmt 6 -evalue 1e-3 -max_target_seqs 5 -max_hsps 1 -num_threads 40 &> blastp.log &
```

**Note**: This takes a while to run, so you may want to run it in the background using `nohup` and `&> blastp.log &` to save the output to a log file. You can also use `-num_threads` to specify the number of threads to use for the BLAST search.

**Note**: The blastp.log file is saved here in a tar.gz file, to decompress it, run the following command:
```bash
tar -zxvf blastp.log.tar.gz
```

To analyze the results, you can use the following command:
```bash
python analyze.py --blast_log_file blastp.log --dataset_dir ../../../data/pfam --output_file blastp_evaluation.json
```

In addition to the JSON file containing the detailed accuracy by sequence and cluster results, the script also produces the following output:
```
Loading Pfam20 families metadata from ../../../data/pfam/pfam20_seed-42/families.json
Loading Pfam20 families metadata from ../../../data/pfam/pfam20_seed-42/families.json
Sequence level mean accuracy: 0.8401727370873345
Cluster level mean accuracy: 0.8401727370873344
Sequence level median accuracy: 1.0
Cluster level median accuracy: 0.95
```
