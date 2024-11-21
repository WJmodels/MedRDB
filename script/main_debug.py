from typing import List
import re
import copy
import traceback
import numpy as np
import multiprocessing

import rdkit
import rdkit.Chem as Chem

ORGANIC_SET = {'B', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I'}

RGROUP_SYMBOLS = ['R', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12',
                  'Ra', 'Rb', 'Rc', 'Rd', 'X', 'Y', 'Z', 'Q', 'A', 'E', 'Ar']

PLACEHOLDER_ATOMS = ["Lv", "Lu", "Nd", "Yb", "At", "Fm", "Er"]


class Substitution(object):
    '''Define common substitutions for chemical shorthand'''
    def __init__(self, abbrvs, smarts, smiles, probability):
        assert type(abbrvs) is list
        self.abbrvs = abbrvs
        self.smarts = smarts
        self.smiles = smiles
        self.probability = probability


SUBSTITUTIONS: List[Substitution] = [
    Substitution(['C13H27'], '[CH2;D2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH3]', "[CH2]CCCCCCCCCCCC", 0.5),

    Substitution(['NO2', 'O2N'], '[N+](=O)[O-]', "[N+](=O)[O-]", 0.5),
    Substitution(['CHO', 'OHC'], '[CH1](=O)', "[CH1](=O)", 0.5),
    Substitution(['CO2Et', 'COOEt'], 'C(=O)[OH0;D2][CH2;D2][CH3]', "[C](=O)OCC", 0.5),

    Substitution(['OAc'], '[OH0;X2]C(=O)[CH3]', "[O]C(=O)C", 0.7),
    Substitution(['NHAc'], '[NH1;D2]C(=O)[CH3]', "[NH]C(=O)C", 0.7),
    Substitution(['Ac'], 'C(=O)[CH3]', "[C](=O)C", 0.1),

    Substitution(['OBz'], '[OH0;D2]C(=O)[cH0]1[cH][cH][cH][cH][cH]1', "[O]C(=O)c1ccccc1", 0.7),  # Benzoyl
    Substitution(['Bz'], 'C(=O)[cH0]1[cH][cH][cH][cH][cH]1', "[C](=O)c1ccccc1", 0.2),  # Benzoyl

    Substitution(['OBn'], '[OH0;D2][CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[O]Cc1ccccc1", 0.7),  # Benzyl
    Substitution(['Bn'], '[CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[CH2]c1ccccc1", 0.2),  # Benzyl

    Substitution(['NHBoc'], '[NH1;D2]C(=O)OC([CH3])([CH3])[CH3]', "[NH1]C(=O)OC(C)(C)C", 0.6),
    Substitution(['NBoc'], '[NH0;D3]C(=O)OC([CH3])([CH3])[CH3]', "[NH1]C(=O)OC(C)(C)C", 0.6),
    Substitution(['Boc'], 'C(=O)OC([CH3])([CH3])[CH3]', "[C](=O)OC(C)(C)C", 0.2),

    Substitution(['Cbm'], 'C(=O)[NH2;D1]', "[C](=O)N", 0.2),
    Substitution(['Cbz'], 'C(=O)OC[cH]1[cH][cH][cH1][cH][cH]1', "[C](=O)OCc1ccccc1", 0.4),
    Substitution(['Cy'], '[CH1;X3]1[CH2][CH2][CH2][CH2][CH2]1', "[CH1]1CCCCC1", 0.3),
    Substitution(['Fmoc'], 'C(=O)O[CH2][CH1]1c([cH1][cH1][cH1][cH1]2)c2c3c1[cH1][cH1][cH1][cH1]3',
                 "[C](=O)OCC1c(cccc2)c2c3c1cccc3", 0.6),
    Substitution(['Mes'], '[cH0]1c([CH3])cc([CH3])cc([CH3])1', "[c]1c(C)cc(C)cc(C)1", 0.5),
    Substitution(['OMs'], '[OH0;D2]S(=O)(=O)[CH3]', "[O]S(=O)(=O)C", 0.7),
    Substitution(['Ms'], 'S(=O)(=O)[CH3]', "[S](=O)(=O)C", 0.2),
    Substitution(['Ph'], '[cH0]1[cH][cH][cH1][cH][cH]1', "[c]1ccccc1", 0.5),
    Substitution(['PMB'], '[CH2;D2][cH0]1[cH1][cH1][cH0](O[CH3])[cH1][cH1]1', "[CH2]c1ccc(OC)cc1", 0.2),
    Substitution(['Py'], '[cH0]1[n;+0][cH1][cH1][cH1][cH1]1', "[c]1ncccc1", 0.1),
    Substitution(['SEM'], '[CH2;D2][CH2][Si]([CH3])([CH3])[CH3]', "[CH2]CSi(C)(C)C", 0.2),
    Substitution(['Suc'], 'C(=O)[CH2][CH2]C(=O)[OH]', "[C](=O)CCC(=O)O", 0.2),
    Substitution(['TBS'], '[Si]([CH3])([CH3])C([CH3])([CH3])[CH3]', "[Si](C)(C)C(C)(C)C", 0.5),
    Substitution(['TBZ'], 'C(=S)[cH]1[cH][cH][cH1][cH][cH]1', "[C](=S)c1ccccc1", 0.2),
    Substitution(['OTf'], '[OH0;D2]S(=O)(=O)C(F)(F)F', "[O]S(=O)(=O)C(F)(F)F", 0.7),
    Substitution(['Tf'], 'S(=O)(=O)C(F)(F)F', "[S](=O)(=O)C(F)(F)F", 0.2),
    Substitution(['TFA'], 'C(=O)C(F)(F)F', "[C](=O)C(F)(F)F", 0.3),
    Substitution(['TMS'], '[Si]([CH3])([CH3])[CH3]', "[Si](C)(C)C", 0.5),
    Substitution(['Ts'], 'S(=O)(=O)c1[cH1][cH1][cH0]([CH3])[cH1][cH1]1', "[S](=O)(=O)c1ccc(C)cc1", 0.6),  # Tos

    # Alkyl chains
    Substitution(['OMe', 'MeO'], '[OH0;D2][CH3;D1]', "[O]C", 0.3),
    Substitution(['SMe', 'MeS'], '[SH0;D2][CH3;D1]', "[S]C", 0.3),
    Substitution(['NMe', 'MeN'], '[N;X3][CH3;D1]', "[NH]C", 0.3),
    Substitution(['Me'], '[CH3;D1]', "[CH3]", 0.1),
    Substitution(['OEt', 'EtO'], '[OH0;D2][CH2;D2][CH3]', "[O]CC", 0.5),
    Substitution(['Et', 'C2H5'], '[CH2;D2][CH3]', "[CH2]C", 0.3),
    Substitution(['Pr', 'nPr', 'n-Pr'], '[CH2;D2][CH2;D2][CH3]', "[CH2]CC", 0.3),
    Substitution(['Bu', 'nBu', 'n-Bu'], '[CH2;D2][CH2;D2][CH2;D2][CH3]', "[CH2]CCC", 0.3),

    # Branched
    Substitution(['iPr', 'i-Pr'], '[CH1;D3]([CH3])[CH3]', "[CH1](C)C", 0.2),
    Substitution(['iBu', 'i-Bu'], '[CH2;D2][CH1;D3]([CH3])[CH3]', "[CH2]C(C)C", 0.2),
    Substitution(['OiBu'], '[OH0;D2][CH2;D2][CH1;D3]([CH3])[CH3]', "[O]CC(C)C", 0.2),
    Substitution(['OtBu'], '[OH0;D2][CH0]([CH3])([CH3])[CH3]', "[O]C(C)(C)C", 0.6),
    Substitution(['tBu', 't-Bu'], '[CH0]([CH3])([CH3])[CH3]', "[C](C)(C)C", 0.3),

    

    # Substitution(['TIPSO', 'OTIPS'], '[OH0][Si]([CH1]([CH3])[CH3])([CH1]([CH3])[CH3])[CH1]([CH3])[CH3]', "[O][Si](C(C)C)(C(C)C)C(C)C", 0.5),

    # Other shorthands (MIGHT NOT WANT ALL OF THESE)
    Substitution(['CF3', 'F3C'], '[CH0;D4](F)(F)F', "[C](F)(F)F", 0.5),
    Substitution(['NCF3', 'F3CN'], '[N;X3][CH0;D4](F)(F)F', "[NH]C(F)(F)F", 0.5),
    Substitution(['OCF3', 'F3CO'], '[OH0;X2][CH0;D4](F)(F)F', "[O]C(F)(F)F", 0.5),
    Substitution(['CCl3'], '[CH0;D4](Cl)(Cl)Cl', "[C](Cl)(Cl)Cl", 0.5),
    Substitution(['CO2H', 'HO2C', 'COOH'], 'C(=O)[OH]', "[C](=O)O", 0.5),  # COOH
    Substitution(['CN', 'NC'], 'C#[ND1]', "[C]#N", 0.5),
    Substitution(['OCH3', 'H3CO'], '[OH0;D2][CH3]', "[O]C", 0.4),
    Substitution(['SO3H'], 'S(=O)(=O)[OH]', "[S](=O)(=O)O", 0.4),

    Substitution(['[ClO2SN]', '[NSO2Cl]', 'ClO2SH'], 'ClS(=O)(=O)[N]', "ClS(=O)(=O)[N]", 0.4),





    Substitution(['MeO2C', 'CO2Me'], 'C(=O)[OH0;D2][CH3;D1]', "[C](=O)OC", 0.5),

    Substitution(['MOMO', 'OMOM'], '[OH0;D2][CH2;D2][OH0;D2][CH3]', "[O]COC", 0.5),
    Substitution(['EOMO', 'OEOM'], '[OH0;D2][CH2;D2][OH0;D2][CH2][CH3]', "[O]COCC", 0.5),
    Substitution(['PMB', 'MPM'], '[CH2;D2][cH0]1[cH1][cH1][cH0](O[CH3])[cH1][cH1]1', "[CH2]c1ccc(OC)cc1", 0.2),
    Substitution(['PMBO', 'OPMB', 'MPMO', 'OMPM'], '[OH0;D3][CH1;D3][cH0]1[cH1][cH1][cH0](O[CH3])[cH1][cH1]1', "[O]Cc1ccc(OC)cc1", 0.5),
    Substitution(['OBND', 'DNBO'], '[O][cH0]1[cH0]([N+](=O)[O-])[cH1][cH0]([N+](=O)[O-])[cH1][cH1]1', "[O]c1c([N+](=O)[O-])cc([N+](=O)[O-])cc1", 0.3),
    Substitution(['BND', 'DNB'], '[cH0]1[cH0]([N+](=O)[O-])[cH1][cH0]([N+](=O)[O-])[cH1][cH1]1', "[c]1c([N+](=O)[O-])cc([N+](=O)[O-])cc1", 0.3),
    Substitution(['PivO','OPiv'], '[OH0;D2]C(=O)C([CH3])([CH3])[CH3]', "[O]C(=O)C(C)(C)C", 0.5),
    Substitution(['Piv'], 'C(=O)C([CH3])([CH3])[CH3]', "C(=O)C(C)(C)C", 0.5),


    # New Addition
    Substitution(['OTIPS', 'TIPSO', 'OSi(iPr)3', '(iPr)3SiO'], '[O][Si]([CH1]([CH3])[CH3])([CH1]([CH3])[CH3])[CH1]([CH3])[CH3]', "[O][Si](C(C)C)(C(C)C)C(C)C", 0.5),
    Substitution(['OTES', 'TESO'], '[O][Si]([CH2][CH3])([CH2][CH3])[CH2][CH3]', "[O][Si](CC)(CC)CC", 0.5),
    Substitution(['TBDPSO','OTBDPS'], '[OH0;D2][Si]([cH0]1[cH][cH][cH][cH][cH]1)([cH0]1[cH][cH][cH][cH][cH]1)[CH0]([CH3])([CH3])[CH3]', "[O][Si](C1=CC=CC=C1)(C2=CC=CC=C2)C(C)(C)C", 0.5),
    Substitution(['OTIPS', 'TIPSO', 'OSi(iPr)3', '(iPr)3SiO'], '[OH0][Si]([CH1]([CH3])[CH3])([CH1]([CH3])[CH3])[CH1]([CH3])[CH3]', "[O][Si](C(C)C)(C(C)C)C(C)C", 0.5),
    Substitution(['TBDMSO','OTBDMS', 'tBuMe2SiO','OSiMe2tBu'], '[OH0;D2][Si]([CH3])(C([CH3])([CH3])[CH3])[CH3]', "[O][Si](C)(C(C)(C)C)C", 0.5),
    Substitution(['PhMe2SiO','OSiMe2Ph'], '[OH0;D2][Si]([CH3])([CH3])[cH0]1[cH][cH][cH][cH][cH]1', "[O][Si](C)(C)C1=CC=CC=C1", 0.5),
    Substitution(['OTBS', 'TBSO'], '[OH0;D2][Si]([CH3])([CH3])C([CH3])([CH3])[CH3]', "[O]Si](C)(C)C(C)(C)C", 0.5),

    Substitution(['Teoc'], '[C](=O)O[CH2][CH2][Si]([CH3])([CH3])[CH3]', "[C](=O)OCC[Si](C)(C)C", 0.5),
    Substitution(['NTeoc', 'TeocN'], '[NH0;D3]C(=O)O[CH2][CH2][Si]([CH3])([CH3])[CH3]', "[N]C(=O)OCC[Si](C)(C)C", 0.5),
    Substitution(['NHTeoc', 'TeocHN'], '[NH1;D2]C(=O)O[CH2][CH2][Si]([CH3])([CH3])[CH3]', "[NH]C(=O)OCC[Si](C)(C)C", 0.5),
    Substitution(['Ns'], '[S](=O)(=O)c1[cH1][cH1][cH0]([N+](=O)[O-])[cH1][cH1]1', "[S](=O)(=O)c1ccc([N+](=O)[O-])cc1", 0.1),


    Substitution(['NHtBu', 'tBuHN'], '[NH1:D2][C]([CH3])([CH3])[CH3]', "[NH]C(C)(C)C", 0.5),
    Substitution(['NBoc', 'BocN'], '[NH0;D3]C(=O)OC([CH3])([CH3])[CH3]', "[N]C(=O)OC(C)(C)C", 0.4),
    Substitution(['NHBoc', 'BocNH'], '[NH1;D2][CH2;D2][OH0]C([CH3])([CH3])[CH3]', "[NH]C(=O)OC(C)(C)C", 0.7),
    Substitution(['NBn', 'BnN'], '[NH0;D3][CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[N]Cc1ccccc1", 0.5),  # NBenzyl
    Substitution(['NMe2'], '[NH0:D3]([CH3])[CH3]', "CN(C)", 0.5),
    Substitution(['CONH2', 'H2NOC'], 'C(=O)[NH2]', "[C](=O)N", 0.5),
    Substitution(['CONHtBu', 'tBuHNOC'], '[C](=O)[NH][CH0]([CH3])([CH3])[CH3]', "[C](=O)[NH]C(C)(C)C",  0.5),


    Substitution(['N2'], '[N+]=[N-]', "[N+]=[N-]",  0.5),
    Substitution(['N3'], '[N:D1]=[N+]=[N-]', "[N]=[N+]=[N-]", 0.5),
    Substitution(['NCS'], '[N:D1]=C=S', "[N]=C=S", 0.5),

    # special elements
    Substitution(['SePh', '[PhSe]'], '[cH]1[cH][cH][cH][cH][cH0]1[Se]', "C2=CC=CC=C2[Se]",  0.5),
    Substitution(['SeMe', 'MeSe'], '[Se][CH3]', "[Se]C",  0.5),
    Substitution(['Sn(Bu)3', 'Bu3Sn', 'SnBu3'], '[Sn]([CH2][CH2][CH2][CH3])([CH2][CH2][CH2][CH3])[CH2][CH2][CH2][CH3]', "[Sn](CCCC)(CCCC)CCCC", 0.5),
    Substitution(['Bpin', 'pinB', 'BPin'], '[BH0]1O[CH0]([CH3])([CH3])[CH0]([CH3])(O1)[CH3]', "[B]1OC(C)(C)C(C)(O1)C",  0.5),
    Substitution(['B(OiPr)2', '(OiPr)2B'], '[B]([O][CH1]([CH3])[CH3])[O][CH1]([CH3])[CH3]', "[B](OC(C)(C))OC(C)(C)",  0.5),
    Substitution(['DIPP'], '[P]([O][CH1]([CH3])[CH3])[O][CH1]([CH3])[CH3]', "[P](OC(C)(C))OC(C)(C)",  0.5)





]

ABBREVIATIONS = {abbrv: sub for sub in SUBSTITUTIONS for abbrv in sub.abbrvs}

VALENCES = {
    "H": [1], "Li": [1], "Be": [2], "B": [3], "C": [4], "N": [3, 5], "O": [2], "F": [1],
    "Na": [1], "Mg": [2], "Al": [3], "Si": [4], "P": [5, 3], "S": [6, 2, 4], "Cl": [1], "K": [1], "Ca": [2],
    "Br": [1], "I": [1]
}

ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]

COLORS = {
    u'c': '0.0,0.75,0.75', u'b': '0.0,0.0,1.0', u'g': '0.0,0.5,0.0', u'y': '0.75,0.75,0',
    u'k': '0.0,0.0,0.0', u'r': '1.0,0.0,0.0', u'm': '0.75,0,0.75'
}

# tokens of condensed formula
FORMULA_REGEX = re.compile(
    '(' + '|'.join(list(ABBREVIATIONS.keys())) + '|R[0-9]*|[A-Z][a-z]+|[A-Z]|[0-9]+|\(|\))')

def _convert_graph_to_smiles(coords, symbols, edges, image=None, debug=False):
    mol = Chem.RWMol()
    n = len(symbols)
    ids = []
    print("symbols:",symbols)
    for i in range(n):
        symbol = symbols[i]
        
        
        symbol = symbol.replace("WOMO","MOMO").replace("TlMSO","TMSO").replace("NHAloc","NHAlloc").replace("Aloc","Alloc").replace("FmodHN", "FmocHN").replace("S@@", "S@@+").replace("TlPSO", "TIPSO").replace("OTlPS", "OTIPS").replace("OTiPS", "OTIPS").replace("TiPSO", "TIPSO").replace("CONH+", "CONH2").replace("BocH", "BocHN").replace("PMB", "PMBO")#.replace("[ClO2SH]", "ClO2SHN")  #替换
        if symbol[0] == '[':
            symbol = symbol[1:-1]
        import ipdb
        ipdb.set_trace()
        if symbol in RGROUP_SYMBOLS:
            atom = Chem.Atom("*")
            if symbol[0] == 'R' and symbol[1:].isdigit():
                atom.SetIsotope(int(symbol[1:]))
            Chem.SetAtomAlias(atom, symbol)
        elif symbol in ABBREVIATIONS:
            atom = Chem.Atom("*")
            Chem.SetAtomAlias(atom, symbol)
        else:
            try:  # try to get SMILES of atom
                atom = Chem.AtomFromSmiles(symbols[i])
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            except:  # otherwise, abbreviation or condensed formula
                atom = Chem.Atom("*")
                Chem.SetAtomAlias(atom, symbol)

        if atom.GetSymbol() == '*':
            atom.SetProp('molFileAlias', symbol)

        idx = mol.AddAtom(atom)
        assert idx == i
        ids.append(idx)
    import ipdb
    ipdb.set_trace()
    for i in range(n):
        for j in range(i + 1, n):
            if edges[i][j] == 1:
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
            elif edges[i][j] == 2:
                mol.AddBond(ids[i], ids[j], Chem.BondType.DOUBLE)
            elif edges[i][j] == 3:
                mol.AddBond(ids[i], ids[j], Chem.BondType.TRIPLE)
            elif edges[i][j] == 4:
                mol.AddBond(ids[i], ids[j], Chem.BondType.AROMATIC)
            elif edges[i][j] == 5:
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(Chem.BondDir.BEGINWEDGE)
            elif edges[i][j] == 6:
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(Chem.BondDir.BEGINDASH)

    pred_smiles = '<invalid>'

    try:
        # TODO: move to an util function
        if image is not None:
            height, width, _ = image.shape
            ratio = width / height
            coords = [[x * ratio * 10, y * 10] for x, y in coords]
        mol = _verify_chirality(mol, coords, symbols, edges, debug)
        # molblock is obtained before expanding func groups, otherwise the expanded group won't have coordinates.
        # TODO: make sure molblock has the abbreviation information
        pred_molblock = Chem.MolToMolBlock(mol)
        pred_smiles, mol = _expand_functional_group(mol, {}, debug)
        success = True
    except Exception as e:
        if debug:
            print(traceback.format_exc())
        pred_molblock = ''
        success = False

    if debug:
        return pred_smiles, pred_molblock, mol, success
    return pred_smiles, pred_molblock, success

if __name__ == '__main__':
    symbols = ['C', '[OMe]', 'C', '[C@H]', '[CO2Me]', 'N', 'C', 'Br', 'C', 'C', 'C', 'C', '[ClO2SH]', 'O', 'O']
    coords = [[0.6535433070866141, 0.1968503937007874], [0.7952755905511811, 0.2755905511811024], [0.5196850393700787, 0.2755905511811024], [0.5196850393700787, 0.4330708661417323], [0.6535433070866141, 0.5118110236220472], [0.3779527559055118, 0.5118110236220472], [0.2283464566929134, 0.44881889763779526], [0.16535433070866143, 0.2992125984251969], [0.11811023622047244, 0.5669291338582677], [0.2047244094488189, 0.7007874015748031], [0.36220472440944884, 0.6692913385826772], [0.48031496062992124, 0.7716535433070866], [0.44881889763779526, 0.9212598425196851], [0.6299212598425197, 0.7244094488188977], [0.6535433070866141, 0.047244094488188976]]
    edges = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 6, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    _convert_graph_to_smiles(coords, symbols, edges, image=None, debug=False)