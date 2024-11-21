from STOUT import translate_forward
from STOUT import translate_reverse
def Convert_SMILES_2_IUPAC(smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C"):
    iupac = translate_forward(smiles)
    return iupac

def Convert_IUPAC_2_SMILES(iupac="1,3,7-trimethylpurine-2,6-dione"):
    smiles = translate_reverse(iupac)
    return smiles