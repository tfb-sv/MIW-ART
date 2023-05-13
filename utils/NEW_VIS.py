##########################################################################################
##########################################################################################
##########################################################################################

cd C:\Users\nural\OneDrive\Masaüstü\ART-Mol

##########################################################################################

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
import numpy as np
# from sklearn import preprocessing
# sns.set_theme(rc={'figure.figsize':(27, 27)}, font_scale=3.5, style="whitegrid")

##########################################################################################
##########################################################################################
##########################################################################################

smi = "CCC#Cc1cccc([C@@]2(c3ccc(OCF)cc3)N=C(N)N(C)C2=O)c1"

weights = [3, 4, 3, 2]   # = is forth and 1 point but no bond will be shown??

######################################

mol = Chem.MolFromSmiles(smi)

weights = np.array(weights)

############################################################################

# contribs = Chem.rdMolDescriptors._CalcCrippenContribs(mol)

# weights = [x for x, y in contribs]

######################################

n_weights = (weights - weights.min()) / (weights.max() - weights.min())   # norm to between [0, 1], no w.min assuming

# n_weights = ((2 * (weights - weights.min())) / (weights.max() - weights.min())) - 1   # norm to between [-1, 1], no w.min assuming

############################################################################

for atom in mol.GetAtoms():
    # atom.SetProp("atomLabel", atom.GetSymbol())
    atom.SetAtomMapNum(atom.GetIdx())
    print(f"{atom.GetIdx()} - {atom.GetSymbol()}\n")

######################################

fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, 
                                                 n_weights, 
                                                 colorMap="jet",   # "CMRmap"
                                                 size=(350, 350),
                                                 contourLines=10   # , kwargs
                                                 )

##########################################################################################
##########################################################################################
##########################################################################################

for atom in mol.GetAtoms():
    # atom.SetAtomMapNum(atom.GetIdx())
    print(f"\n{atom.GetIdx()} - {atom.GetSymbol()}")

##########################################################################################

# OLUR BU İŞ! SMILES'DAKİ KARAKTER SIRALAMASINA GÖRE NUMARALANDIRIYOR RDKIT DE!
# YALNIZ SMILES'LARI ARINDIRMAK LAZIM GEREKSİZ ŞEYLERDEN MESELA BAĞLAR VS.
# HALKALI YAPILAR, UCU AÇIK DALLAR FALAN HER ŞEY 1E1 UYUYOR !!! 
# O ZAMAN GELSİN GÖRSELLER !! yes





































































