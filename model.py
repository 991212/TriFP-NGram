import torch

from module_drug import *
from module_protein import *

class DTANet(nn.Module):
    def __init__(self,hp, block_num, vocab_protein_size, vocab_drug_size, out_dim=1):
        super().__init__()
        self.protein_encoder = Protein_Seq_Representation(block_num, vocab_protein_size, hp.embedding_size,out_dim=hp.module_out_dim)
        self.kmer_encoder = Protein_Kmer_Representation(hp)

        self.drug_graph_encoder = Drug_Graph_Representation(num_input_features=22, out_dim=hp.module_out_dim, block_config=[8, 8, 8], bn_sizes=[2, 2, 2])

        self.drug_mixedFP_encoder = Drug_MixedFP_Representation(hp)
        self.classifier = nn.Sequential(
            nn.Linear(hp.module_out_dim * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim)
        )

    def forward(self, data):
        '''protein feature extraction module'''
        # target: torch.Size([512, 900])
        target = data.target
        protein_sequence_feature = self.protein_encoder(target)

        '''protein k-mer feature extraction module'''
        kmer = data.kmer
        protein_kmer_feature = self.kmer_encoder(kmer)

        '''drug molecule feature extraction module'''
        drug_graph_feature = self.drug_graph_encoder(data)

        '''Drug mixed fingerprint (MACCS、PubChem、Pharmacophore ErG) feature extraction module'''
        mixedFP = data.mixedFP
        drug_mixedFP_feature = self.drug_mixedFP_encoder(mixedFP)

        drug_protein_feature = torch.cat([drug_graph_feature,drug_mixedFP_feature,protein_sequence_feature,protein_kmer_feature],dim=-1)

        output = self.classifier(drug_protein_feature)

        return output


