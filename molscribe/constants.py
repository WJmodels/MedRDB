from typing import List
import re

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
    Substitution(['CHO', 'OHC'], '[CH1](=O)', "[CH1](=O)", 0.5),
    Substitution(['CF3', 'F3C'], '[CH0;D4](F)(F)F', "[C](F)(F)F", 0.5),
    Substitution(['CN', 'NC'], '[N]#[C]', "[C]#N", 0.5),
    Substitution(['CCl3', 'Cl3C'], '[CH0;D4](Cl)(Cl)CL', "[C](Cl)(Cl)Cl", 0.5),
    Substitution(['CO2H', 'HO2C', 'COOH'], 'C(=O)[OH]', "[C](=O)O", 0.5),
    Substitution(['MsOH', 'CH3SO3H'], 'OS(=O)(=O)[CH3]', "OS(=O)(=O)C", 0.7),
    Substitution(['TsOH', 'p-TsOH', 'pTsOH'], 'OS(=O)(=O)c1[cH1][cH1][cH0]([CH3])[cH1][cH1]1', "OS(=O)(=O)c1ccc(C)cc1", 0.6),
    Substitution(['NO2', 'O2N'], '[N+](=O)[O-]', "[N+](=O)[O-]", 0.5),
    Substitution(['Ph'], '[cH0]1[cH][cH][cH1][cH][cH]1', "[c]1ccccc1", 0.5),
    Substitution(['Py'], '[cH0]1[n;+0][cH1][cH1][cH1][cH1]1', "[c]1ncccc1", 0.1),
    Substitution(['N2'], '[N+]=[N-]', "[N+]=[N-]",  0.5),
    Substitution(['N3'], '[N:D1]=[N+]=[N-]', "[N]=[N+]=[N-]", 0.5),
    
    # Protecting Groups
    Substitution(['Ac'], 'C(=O)[CH3]', "[C](=O)C", 0.1),
    Substitution(['Alloc'], 'C(=O)[OH0;D2][CH2;D2][CH1;D2](=[CH2])', "C(=O)OCC=C", 0.5),
    Substitution(['Bn'], '[CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[CH2]c1ccccc1", 0.2),  # Benzyl
    Substitution(['Bz'], 'C(=O)[cH0]1[cH][cH][cH][cH][cH]1', "[C](=O)c1ccccc1", 0.2),  # Benzoyl
    Substitution(['Boc', 'BOC'], 'C(=O)OC([CH3])([CH3])[CH3]', "[C](=O)OC(C)(C)C", 0.2),
    Substitution(['Cbm'], 'C(=O)[NH2;D1]', "[C](=O)N", 0.2),
    Substitution(['Cbz', 'CBz'], 'C(=O)OC[cH]1[cH][cH][cH1][cH][cH]1', "[C](=O)OCc1ccccc1", 0.4),
    Substitution(['Cy'], '[CH1;X3]1[CH2][CH2][CH2][CH2][CH2]1', "[CH1]1CCCCC1", 0.3),
    Substitution(['DMB'], '[CH2;D2]C1=CC(OC)=CC=C1OC', "[CH2]C1=CC(OC)=CC=C1OC", 0.5),
    Substitution(['Fmoc'], 'C(=O)O[CH2][CH1]1c([cH1][cH1][cH1][cH1]2)c2c3c1[cH1][cH1][cH1][cH1]3',
                 "[C](=O)OCC1c(cccc2)c2c3c1cccc3", 0.6),
    Substitution(['Mes'], '[cH0]1c([CH3])cc([CH3])cc([CH3])1', "[c]1c(C)cc(C)cc(C)1", 0.5),
    Substitution(['Ms'], 'S(=O)(=O)[CH3]', "[S](=O)(=O)C", 0.2),
    Substitution(['PMB', 'MPM'], '[CH2;D2][cH0]1[cH1][cH1][cH0](O[CH3])[cH1][cH1]1', "[CH2]c1ccc(OC)cc1", 0.2),
    Substitution(['SEM'], '[CH2;D2][CH2][Si]([CH3])([CH3])[CH3]', "[CH2]CSi(C)(C)C", 0.2),
    Substitution(['Suc'], 'C(=O)[CH2][CH2]C(=O)[OH]', "[C](=O)CCC(=O)O", 0.2),
    Substitution(['TBZ'], 'C(=S)[cH0]1[cH][cH][cH1][cH][cH]1', "[C](=S)c1ccccc1", 0.2),
    Substitution(['Tf'], 'S(=O)(=O)C(F)(F)F', "[S](=O)(=O)C(F)(F)F", 0.2),
    Substitution(['TFA'], 'C(=O)C(F)(F)F', "[C](=O)C(F)(F)F", 0.3),
    Substitution(['TMS', 'SiMe3', 'Me3Si', 'Si(CH3)3'], '[Si]([CH3])([CH3])[CH3]', "[Si](C)(C)C", 0.5),
    Substitution(['Ts'], 'S(=O)(=O)c1[cH1][cH1][cH0]([CH3])[cH1][cH1]1', "[S](=O)(=O)c1ccc(C)cc1", 0.6),
    Substitution(['TIPS'], '[Si]([CH1]([CH3])[CH3])([CH1]([CH3])[CH3])[CH1]([CH3])[CH3]', "[Si](C(C)C)(C(C)C)C(C)C", 0.5),
    Substitution(['MOM', 'MOM'], '[CH2;D2][OH0;D2][CH3]', "[CH2]OC", 0.5),
    Substitution(['Tr', 'CPh3', 'Trt', '(Ph)3C', 'Ph3C'], '[C]([cH0]1[cH][cH][cH][cH][cH]1)([cH0]1[cH][cH][cH][cH][cH]1)([cH0]1[cH][cH][cH][cH][cH]1)', "[C](c1ccccc1)(c1ccccc1)(c1ccccc1)", 0.5),
    Substitution(['SiMe2Ph','PhMe2Si', 'Si(CH3)2Ph', 'Ph(CH3)2Si'], '[Si]([CH3])([CH3])[cH0]1[cH][cH][cH][cH][cH]1', "[Si](C)(C)c1ccccc1", 0.5),
    Substitution(['Troc', 'Toc'], '[C](=O)[O][CH2][C](Cl)(Cl)Cl', "[C](=O)OCC(Cl)(Cl)Cl", 0.5),
    Substitution(['THP'], '[CH]1CCCCO1', "[CH]1CCCCO1", 0.5),
    Substitution(['BND', 'DNB'], '[cH0]1[cH0]([N+](=O)[O-])[cH1][cH0]([N+](=O)[O-])[cH1][cH1]1', "[c]1c([N+](=O)[O-])cc([N+](=O)[O-])cc1", 0.3),
    Substitution(['Piv'], 'C(=O)C([CH3])([CH3])[CH3]', "C(=O)C(C)(C)C", 0.5),
    Substitution(['PMP'], '[cH0]1[cH1][cH1][cH0](O[CH3])[cH1][cH1]1', "[c]1ccc(OC)cc1",  0.5),
    Substitution(['TBS', 'TBDMS','TBDMS', 'tBuMe2Si','SiMe2tBu'], '[Si]([CH3])(C([CH3])([CH3])[CH3])[CH3]', "[Si](C)(C(C)(C)C)C", 0.5),
    Substitution(['TES', 'Si(Et)3', 'Et3Si', 'SiEt3'], '[Si]([CH2][CH3])([CH2][CH3])[CH2][CH3]', "[Si](CC)(CC)CC", 0.5),
    Substitution(['TBDPS', 'TBDPS'], '[Si]([cH0]1[cH][cH][cH][cH][cH]1)([cH0]1[cH][cH][cH][cH][cH]1)[CH0]([CH3])([CH3])[CH3]', "[Si](C1=CC=CC=C1)(C2=CC=CC=C2)C(C)(C)C", 0.5),
    Substitution(['SiMe2F','FMe2Si'], '[Si]([CH3])([CH3])F', "[Si](C)(C)F", 0.5),
    Substitution(['Bn3Si','SiBn3'], '[Si]([CH2;D2][cH0]1[cH][cH][cH][cH][cH]1)([CH2;D2][cH0]1[cH][cH][cH][cH][cH]1)[CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[Si](Cc1ccccc1)(CCc1ccccc1)Cc1ccccc1", 0.5),
    Substitution(['SEM'], '[CH2;D2]O[CH2][CH2][Si]([CH3])([CH3])[CH3]', "[CH2]O[CH2]C[Si](C)(C)C", 0.2),
    Substitution(['Teoc'], '[C](=O)O[CH2][CH2][Si]([CH3])([CH3])[CH3]', "[C](=O)OCC[Si](C)(C)C", 0.5),
    Substitution(['Ns'], '[S](=O)(=O)c1[cH1][cH1][cH0]([N+](=O)[O-])[cH1][cH1]1', "[S](=O)(=O)c1ccc([N+](=O)[O-])cc1", 0.1),
    
    # Ester Group
    Substitution(['CO2Et', 'COOEt', 'EtO2C', 'EtOOC'], 'C(=O)[OH0;D2][CH2;D2][CH3]', "[C](=O)OCC", 0.5),
    Substitution(['MeO2C', 'CO2Me', 'COOMe', 'MeOOC', 'CO2CH3', 'H3CO2C', 'MeO2'], 'C(=O)[OH0;D2][CH3;D1]', "[C](=O)OC", 0.5),
    Substitution(['OCOMe'], '[O]C(=O)[CH3;D1]', "[O]C(=O)C", 0.5),
    Substitution(['CO2tBu', 'CO2But', 'CO2t-Bu', 'COOtBu', 't-BuO2C', 'tBuO2C', 'ButO2C'], 'C(=O)[OH0][CH0]([CH3])([CH3])([CH3])', "C(=O)OC(C)(C)(C)", 0.5),
    Substitution(['CO2Bu','BuO2C'], 'C(=O)[OH0][CH2][CH2][CH2][CH3]', "C(=O)OCCCC", 0.5),
    Substitution(['CO2CF3'], 'C(=O)[OH0][CH0;D4](F)(F)F', "C(=O)O[CH0;D4](F)(F)F", 0.5),
    Substitution(['CO2allyl', 'CO2Allyl'], 'C(=O)[OH0;D2][CH2][CH2;D2]=[CH2]', "[C](=O)OCC=C", 0.5),
    Substitution(['CO2Ph', 'PhO2C', 'Ph(O)CO', 'OC(O)Ph', 'PhCO2', 'PhCOO'], 'C(=O)[OH0;D2][cH0]1[cH][cH][cH][cH][cH]1', "[C](=O)Oc1ccccc1", 0.5),
    Substitution(['PhOC(S)O'], '[O]C(=S)[OH0;D2][cH0]1[cH][cH][cH][cH][cH]1', "[O][C](=S)Oc1ccccc1", 0.5),
    Substitution(['CO2Bn', 'BnO2C'], '[C](=O)[OH0;D2][CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[C](=O)OCc1ccccc1", 0.5),
    Substitution(['SO2Ph', 'PhO2S'], 'S(=O)(=O)[cH0]1[cH][cH][cH][cH][cH]1', "S(=O)(=O)c1ccccc1", 0.4),
    Substitution(['SO2Me', 'MeO2S', 'SO2CH3'], 'S(=O)(=O)[CH3]', "S(=O)(=O)C", 0.4),
    Substitution(['SO2Mes'], 'S(=O)(=O)[cH0]1[cH0]([CH3])[cH][cH0]([CH3])[cH][cH0]([CH3])1', "S(=O)(=O)c1c(C)cc(C)cc(C)1", 0.4),
    
    # O-substituted abbreviation
    Substitution(['OAc', 'AcO'], '[OH0;X2]C(=O)[CH3]', "[O]C(=O)C", 0.7),
    Substitution(['OAllyl'], '[O][OH0;D2][CH2;D2]=[CH2]', "OCC=C", 0.5),
    Substitution(['OBn', 'BnO'], '[OH0;D2][CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[O]Cc1ccccc1", 0.7),
    Substitution(['OBs', 'BsO'], '[OH0;D2]S(=O)(=O)[cH0]1[cH][cH][cH][cH][cH]1', "[O]S(=O)(=O)c1ccccc1", 0.7),
    Substitution(['OBut', 'OtBu', 'Ot-Bu', 't-BuO', 'tBuO'], '[OH0][CH0]([CH3])([CH3])([CH3])', "[O]C(C)(C)(C)", 0.5),
    Substitution(['OBz', 'BzO'], '[OH0;D2]C(=O)[cH0]1[cH][cH][cH][cH][cH]1', "[O]C(=O)c1ccccc1", 0.7),
    Substitution(['OCbz', 'OCBz'], '[O]C(=O)OC[cH]1[cH][cH][cH1][cH][cH]1', "[O]C(=O)OCc1ccccc1", 0.4),
    Substitution(['OCF3', 'F3CO'], '[OH0;X2][CH0;D4](F)(F)F', "[O]C(F)(F)F", 0.5),
    Substitution(['OCH2Br'], '[OH0;D2][CH2;D2]Br', "[O]CBr", 0.3),
    Substitution(['OCHF2', 'F2HCO'], '[OH0;D2][CH1;D3](F)F', "[O]C(F)F", 0.3),
    Substitution(['OEt', 'EtO'], '[OH0;D2][CH2;D2][CH3]', "[O]CC", 0.5),
    Substitution(['OFmoc', 'FmocO', 'Fmoc-O'], '[O]C(=O)O[CH2][CH1]1c([cH1][cH1][cH1][cH1]2)c2c3c1[cH1][cH1][cH1][cH1]3',
                 "[O]C(=O)OCC1c(cccc2)c2c3c1cccc3", 0.6),
    Substitution(['OiBu', 'Oi-Bu'], '[OH0;D2][CH2;D2][CH1;D3]([CH3])[CH3]', "[O]CC(C)C", 0.2),
    Substitution(['OiPr', 'Oi-Pr', 'OPr', 'iPrO'], '[O][CH1;D3]([CH3])[CH3]', "[O]C(C)C", 0.2),
    Substitution(['OLev', 'LevO'], '[O]C(=O)[CH2][CH2]C(=O)[CH3]', "[O]C(=O)CCC(=O)C", 0.3),
    Substitution(['OMs', 'MsO'], '[OH0;D2]S(=O)(=O)[CH3]', "[O]S(=O)(=O)C", 0.7),
    Substitution(['OTf', 'TfO', 'TiO'], '[OH0;D2]S(=O)(=O)C(F)(F)F', "[O]S(=O)(=O)C(F)(F)F", 0.7),
    Substitution(['OTFA'], '[O]C(=O)C(F)(F)F', "[O]C(=O)C(F)(F)F", 0.3),
    Substitution(['OMe', 'MeO', 'OCH3', 'H3CO'], '[OH0;D2][CH3;D1]', "[O]C", 0.3),
    Substitution(['OnBu', 'nBuO'], '[OH0;D2][CH2;D2][CH2][CH2][CH3]', "[O]CCCC", 0.2),
    Substitution(['OPOM', 'POMO'], '[OH0;D2][CH2]OC(=O)[CH0]([CH3])([CH3])[CH3]', "[O]COC(=O)C(C)(C)C", 0.5),
    Substitution(['OtBu', 'Ot-Bu'], '[OH0;D2][CH0]([CH3])([CH3])[CH3]', "[O]C(C)(C)C", 0.6),
    Substitution(['Br(CH2)4O'], '[O][CH2][CH2][CH2][CH2]Br', "[O]CCCCBr", 0.4),
    Substitution(['OBOM', 'BOMO'], '[O][CH2][O][CH2][cH0]1[cH][cH][cH][cH][cH]1', "[O]COCc1ccccc1", 0.5),
    Substitution(['MOMO', 'OMOM'], '[OH0;D2][CH2;D2][OH0;D2][CH3]', "[O]COC", 0.5),
    Substitution(['MEMO', 'OMEM'], '[OH0;D2][CH2;D2][OH0][CH2][CH2][OH0][CH3]', "[O]COCCOC", 0.5),
    Substitution(['EOMO', 'OEOM'], '[OH0;D2][CH2;D2][OH0;D2][CH2][CH3]', "[O]COCC", 0.5),
    Substitution(['OTr', 'TrO'], '[OH0;D2]C([cH0]1[cH][cH][cH][cH][cH]1)([cH0]1[cH][cH][cH][cH][cH]1)([cH0]1[cH][cH][cH][cH][cH]1)', "[O]C(c1ccccc1)(c1ccccc1)(c1ccccc1)", 0.5),
    Substitution(['OTHP', 'THPO'], '[OH0;D2]C1CCCCO1', "[O]C1CCCCO1", 0.5),
    Substitution(['PMBO', 'OPMB', 'MPMO', 'OMPM'], '[OH0;D2][CH2][cH0]1[cH1][cH1][cH0](O[CH3])[cH1][cH1]1', "[O]Cc1ccc(OC)cc1", 0.5),
    Substitution(['PNBO', 'OPNB'], '[OH0;D2]C(=O)[cH0]1[cH1][cH1][cH0]([N+](=O)[O-])[cH1][cH1]1', "[O]C(=O)c1ccc([N+](=O)[O-])cc1", 0.5),
    Substitution(['OBND', 'DNBO'], '[O][cH0]1[cH0]([N+](=O)[O-])[cH1][cH0]([N+](=O)[O-])[cH1][cH1]1', "[O]c1c([N+](=O)[O-])cc([N+](=O)[O-])cc1", 0.3),
    Substitution(['PivO','OPiv'], '[OH0;D2]C(=O)C([CH3])([CH3])[CH3]', "[O]C(=O)C(C)(C)C", 0.5),
    Substitution(['OPMP', 'PMPO'], '[OH0][cH0]1[cH1][cH1][cH0](O[CH3])[cH1][cH1]1', "[O]c1ccc(OC)cc1",  0.5),
    Substitution(['OTs', 'TsO', 'O-Ts'], '[OH0;D2]S(=O)(=O)c1[cH1][cH1][cH0]([CH3])[cH1][cH1]1', "[O]S(=O)(=O)c1ccc(C)cc1", 0.6),
    Substitution(['OPh', 'PhO'], '[OH0;D2][cH0]1[cH][cH][cH1][cH][cH]1', "[O]c1ccccc1", 0.5),
    Substitution(['OPfp'], '[OH0;D2][cH0]1[cH0](F)[cH0](F)[cH0](F)[cH0](F)[cH0]1(F)', "[O]c1c(F)c(F)c(F)c(F)c1(F)", 0.5),
    Substitution(['TMSO', 'OTMS', 'OSiMe3', 'Me3SiO'], '[O][Si]([CH3])([CH3])[CH3]', "[O][Si](C)(C)C", 0.5),
    Substitution(['OTIPS', 'TIPSO', 'OSi(iPr)3', '(iPr)3SiO', 'OSi(Pr)3', '(Pr)3SiO', 'TlPO'], '[O][Si]([CH1]([CH3])[CH3])([CH1]([CH3])[CH3])[CH1]([CH3])[CH3]', "[O][Si](C(C)C)(C(C)C)C(C)C", 0.5),
    Substitution(['OTES', 'TESO', 'Et3SiO', 'OSiEt3', 'OSiEr3'], '[O][Si]([CH2][CH3])([CH2][CH3])[CH2][CH3]', "[O][Si](CC)(CC)CC", 0.5),
    Substitution(['TBDPSO','OTBDPS'], '[OH0;D2][Si]([cH0]1[cH][cH][cH][cH][cH]1)([cH0]1[cH][cH][cH][cH][cH]1)[CH0]([CH3])([CH3])[CH3]', "[O][Si](C1=CC=CC=C1)(C2=CC=CC=C2)C(C)(C)C", 0.5),
    Substitution(['OTBS', 'TBSO', 'TBDMSO','OTBDMS', 'tBuMe2SiO','OSiMe2tBu', 'OSiMe2But'], '[OH0;D2][Si]([CH3])(C([CH3])([CH3])[CH3])[CH3]', "[O][Si](C)(C(C)(C)C)C", 0.5),
    Substitution(['OSiMe2Ph','PhMe2SiO', 'OSi(CH3)2Ph', 'Ph(CH3)2SiO'], '[OH0;D2][Si]([CH3])([CH3])[cH0]1[cH][cH][cH][cH][cH]1', "[O][Si](C)(C)C1=CC=CC=C1", 0.5),
    Substitution(['OSiMe2F','FMe2SiO'], '[OH0;D2][Si]([CH3])([CH3])F', "[O][Si](C)(C)F", 0.5),
    Substitution(['PhMe2SiO','OSiMe2Ph'], '[OH0;D2][Si]([CH3])([CH3])[cH0]1[cH][cH][cH][cH][cH]1', "[O][Si](C)(C)C1=CC=CC=C1", 0.5),
    Substitution(['OCb'], '[O]C(=O)[N]([CH]([CH3])[CH3])[CH]([CH3])[CH3]', "[O]C(=O)[N](C(C)C)C(C)C", 0.5),
    Substitution(['OSu'], '[O][N]1[C](=O)[CH2][CH2][C](=O)1', "[O]N1C(=O)CCC(=O)1", 0.5),

    # N-substituted abbreviation
    Substitution(['NAc', 'AcN'], '[NH0;D3]C(=O)[CH3]', "[N]C(=O)C", 0.7),
    Substitution(['NBn', 'BnN'], '[NH0;D3][CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[N]Cc1ccccc1", 0.5),
    Substitution(['NBoc'], '[NH0;D3]C(=O)OC([CH3])([CH3])[CH3]', "[N]C(=O)OC(C)(C)C", 0.6),
    Substitution(['NBoc2', '(Boc)2N', 'N(Boc)2'], '[NH0;D3](C(=O)OC([CH3])([CH3])[CH3])C(=O)OC([CH3])([CH3])[CH3]', "[N](C(=O)OC(C)(C)C)C(=O)OC(C)(C)C", 0.6),
    Substitution(['NBz', 'BzN'], '[N]C(=O)[cH0]1[cH][cH][cH][cH][cH]1', "[N]C(=O)c1ccccc1", 0.7),
    Substitution(['NCbz', 'NCBz'], '[N]C(=O)OC[cH]1[cH][cH][cH1][cH][cH]1', "[N]C(=O)OCc1ccccc1", 0.4),
    Substitution(['NHAc', 'AcHN'], '[NH1;D2]C(=O)[CH3]', "[NH]C(=O)C", 0.7),
    Substitution(['NHBn', 'BnHN'], '[NH1;D2][CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[NH]Cc1ccccc1", 0.5),
    Substitution(['NHBz', 'BzHN'], '[NH]C(=O)[cH0]1[cH][cH][cH][cH][cH]1', "[NH]C(=O)c1ccccc1", 0.7),
    Substitution(['NHCbz', 'NHCBz', 'CBZNH', 'CBzHN', 'CbzHN'], '[NH]C(=O)OC[cH]1[cH][cH][cH1][cH][cH]1', "[NH]C(=O)OCc1ccccc1", 0.4),
    Substitution(['NHFmoc', 'FmocHN'], '[NH]C(=O)O[CH2][CH1]1c([cH1][cH1][cH1][cH1]2)c2c3c1[cH1][cH1][cH1][cH1]3',
                 "[NH]C(=O)OCC1c(cccc2)c2c3c1cccc3", 0.6),
    Substitution(['NHMoc', 'MocHN'], '[NH1;D2]C(=O)O[CH3]', "[NH1]C(=O)OC", 0.6),
    Substitution(['NHMs', 'MsHN'], '[NH1;D2]S(=O)(=O)[CH3]', "[NH]S(=O)(=O)C", 0.7),
    Substitution(['NHTf', 'TfHN'], '[NH]S(=O)(=O)C(F)(F)F', "[NH]S(=O)(=O)C(F)(F)F", 0.2),
    Substitution(['NMe', 'MeN'], '[N;X3][CH3;D1]', "[N]C", 0.3),
    Substitution(['NMe2', 'Me2N', 'N(Me)2'], '[N;X3]([CH3;D1])[CH3;D1]', "[N](C)C", 0.3),
    Substitution(['NMe3I', 'IMe3N'], '[N+]([CH3;D1])([CH3;D1])[CH3;D1][I-]', "[N+](C)(C)C[I-]", 0.3),
    Substitution(['NEt2', 'Et2N'], '[N;X3]([CH2;D2][CH3;D1])[CH2;D2][CH3;D1]', "[N](CC)CC", 0.3),
    Substitution(['NCF3', 'F3CN'], '[N;X3][CH0;D4](F)(F)F', "[NH]C(F)(F)F", 0.5),
    Substitution(['NNHSO2Ar'], '[N][NH]S(=O)(=O)[cH0]1[cH][cH][cH1][cH][cH]1', "[N][NH]S(=O)(=O)c1ccccc1", 0.4),
    Substitution(['NHTs', 'TsHN'], '[NH]S(=O)(=O)c1[cH1][cH1][cH0]([CH3])[cH1][cH1]1', "[NH]S(=O)(=O)c1ccc(C)cc1", 0.6),
    Substitution(['NNHTs'], '[N][NH]S(=O)(=O)c1[cH1][cH1][cH0]([CH3])[cH1][cH1]1', "[N][NH]S(=O)(=O)c1ccc(C)cc1", 0.6),
    Substitution(['NHPh', 'PhHN'], '[NH][cH0]1[cH][cH][cH1][cH][cH]1', "[NH]c1ccccc1", 0.5),
    Substitution(['NTs', 'TsN'], '[N]S(=O)(=O)c1[cH1][cH1][cH0]([CH3])[cH1][cH1]1', "[N]S(=O)(=O)c1ccc(C)cc1", 0.6),
    Substitution(['NtBu'], '[N][CH0]([CH3])([CH3])([CH3])', "[N]C(C)(C)(C)", 0.4),
    Substitution(['NHTrt', 'TrtHN'], '[NH1;D2]C([cH0]1[cH][cH][cH][cH][cH]1)([cH0]1[cH][cH][cH][cH][cH]1)([cH0]1[cH][cH][cH][cH][cH]1)', "[NH]C(c1ccccc1)(c1ccccc1)(c1ccccc1)", 0.5),
    Substitution(['NHTroc', 'TrocHN'], '[NH]C(=O)[O][CH2][C](Cl)(Cl)Cl', "[NH]C(=O)OCC(Cl)(Cl)Cl", 0.5),
    Substitution(['ClO2SN', 'NSO2Cl', 'ClO2SH'], '[N]S(=O)(=O)Cl', "[N]S(=O)(=O)Cl", 0.4),
    Substitution(['NHDMB','DMBNH'], '[NH;D2][CH2;D2]C1=CC(OC)=CC=C1OC', "[NH]CC1=CC(OC)=CC=C1OC", 0.5),
    Substitution(['NHAlloc','AllocHN'], '[NH1;D2]C(=O)[OH0;D2][CH2;D2][CH1;D2](=[CH2])', "[NH1]C(=O)OCC=C", 0.5),
    Substitution(['NAlloc','AllocN'], '[NH0;D3]C(=O)[OH0;D2][CH2;D2][CH1;D2](=[CH2])', "[N]C(=O)OCC=C", 0.5),
    Substitution(['MeO2CHN', 'NHCO2Me', 'NHCOOMe', 'MeOOCHN'], '[NH]C(=O)[OH0;D2][CH3;D1]', "[NH]C(=O)OC", 0.5),
    Substitution(['MeO2CN', 'NCO2Me', 'NCOOMe', 'MeOOCN'], '[N]C(=O)[OH0;D2][CH3;D1]', "[N]C(=O)OC", 0.5),
    Substitution(['BnO2CHN', 'NHCO2Bn'], '[NH]C(=O)[OH0;D2][CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[NH]C(=O)OCc1ccccc1", 0.5),
    Substitution(['BnON', 'NOBn'], '[N]O[CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[N]OCc1ccccc1", 0.5),
    Substitution(['BnOHN', 'NHOBn'], '[NH]O[CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[NH]OCc1ccccc1", 0.5),
    Substitution(['NCOMe'], '[N]C(=O)[CH3;D1]', "[N]C(=O)C", 0.5),
    Substitution(['PMBN', 'NPMB', 'MPMN', 'NMPM'], '[N][CH2][cH0]1[cH1][cH1][cH0](O[CH3])[cH1][cH1]1', "[N]Cc1ccc(OC)cc1", 0.5),
    Substitution(['PMBHN', 'NHPMB', 'MPMHN', 'NHMPM'], '[NH][CH2][cH0]1[cH1][cH1][cH0](O[CH3])[cH1][cH1]1', "[NH]Cc1ccc(OC)cc1", 0.5),
    Substitution(['NTIPS', 'TIPSN'], '[N][Si]([CH1]([CH3])[CH3])([CH1]([CH3])[CH3])[CH1]([CH3])[CH3]', "[N][Si](C(C)C)(C(C)C)C(C)C", 0.5),
    Substitution(['NTBS', 'TBSN'], '[N][Si]([CH3])([CH3])C([CH3])([CH3])[CH3]', "[N][Si](C)(C)C(C)(C)C", 0.5),
    Substitution(['NHTBS', 'TBSNH'], '[NH][Si]([CH3])([CH3])C([CH3])([CH3])[CH3]', "[NH][Si](C)(C)C(C)(C)C", 0.5),
    Substitution(['NSEM', 'SEMN'], '[NH0;D3][CH2]O[CH2][CH2][Si]([CH3])([CH3])[CH3]', "[N][CH2]O[CH2]C[Si](C)(C)C", 0.5),
    Substitution(['NNH2'], '[N][NH2]', "[N]N", 0.5),
    Substitution(['NHtBu', 'tBuHN'], '[NH1:D2][C]([CH3])([CH3])[CH3]', "[NH]C(C)(C)C", 0.5),
    Substitution(['NBoc', 'BocN'], '[NH0;D3]C(=O)OC([CH3])([CH3])[CH3]', "[N]C(=O)OC(C)(C)C", 0.4),
    Substitution(['NMeBoc', 'BocMeN'], '[NH0;D3]([CH3])C(=O)OC([CH3])([CH3])[CH3]', "[N](C)C(=O)OC(C)(C)C", 0.4),
    Substitution(['NHBoc', 'BocHN', 'BocNH', 'BodH', 'BocH'], '[NH1;D2]C(=O)[OH0]C([CH3])([CH3])[CH3]', "[NH]C(=O)OC(C)(C)C", 0.7),
    Substitution(['NNs', 'NsN'], '[N]S(=O)(=O)c1[cH1][cH1][cH0]([N+](=O)[O-])[cH1][cH1]1', "[N]S(=O)(=O)c1ccc([N+](=O)[O-])cc1", 0.1),
    Substitution(['NHNs', 'NsHN'], '[NH]S(=O)(=O)c1[cH1][cH1][cH0]([N+](=O)[O-])[cH1][cH1]1', "[NH]S(=O)(=O)c1ccc([N+](=O)[O-])cc1", 0.1),
    Substitution(['NHMe', 'MeHN'], '[NH1:D2][CH3]', "[NH](C)", 0.5),
    Substitution(['NHSOtBu'], '[NH1;D2]S(=O)[C]([CH3])([CH3])[CH3]', "[NH]S(=O)C(C)(C)C", 0.5),
    Substitution(['NTeoc', 'TeocN'], '[NH0;D3]C(=O)O[CH2][CH2][Si]([CH3])([CH3])[CH3]', "[N]C(=O)OCC[Si](C)(C)C", 0.5),
    Substitution(['NBnTeoc', 'TeocBnN'], '[NH0;D3]([CH2;D2][cH0]1[cH][cH][cH][cH][cH]1)C(=O)O[CH2][CH2][Si]([CH3])([CH3])[CH3]', "[N]([CH2;D2][cH0]1[cH][cH][cH][cH][cH]1)C(=O)OCC[Si](C)(C)C", 0.5),
    Substitution(['NHTeoc', 'TeocHN'], '[NH1;D2]C(=O)O[CH2][CH2][Si]([CH3])([CH3])[CH3]', "[NH]C(=O)OCC[Si](C)(C)C", 0.5),
    Substitution(['PhthN', 'NPhth'], '[N]1C(=O)(C2=CC=CC=C2C(=O)1)', "[N]1C(=O)(C2=CC=CC=C2C(=O)1)", 0.5),

    # P-substituted abbreviation
    Substitution(['PPh3', 'Ph3P'], '[P+]([cH0]1[cH][cH][cH1][cH][cH]1)([cH0]1[cH][cH][cH1][cH][cH]1)[cH0]1[cH][cH][cH1][cH][cH]1', "[P+](c1ccccc1)(c1ccccc1)c1ccccc1",  0.5),
    Substitution(['PPh2', 'Ph2P'], '[P]([cH0]1[cH][cH][cH1][cH][cH]1)[cH0]1[cH][cH][cH1][cH][cH]1', "[P](c1ccccc1)c1ccccc1",  0.5),
    Substitution(['POPh2', 'Ph2OP'], '[P](=O)([cH0]1[cH][cH][cH1][cH][cH]1)[cH0]1[cH][cH][cH1][cH][cH]1', "[P](=O)(c1ccccc1)c1ccccc1",  0.5),
    Substitution(['(EtO)2(O)P', 'P(O)(OEt)2'], '[P](=O)(O[CH2][CH3])O[CH2][CH3]', "[P](=O)(OCC)OCC",  0.5),
    Substitution(['P(OEt)2'], '[P](O[CH2][CH3])O[CH2][CH3]', "[P](OCC)OCC",  0.5),
    Substitution(['PO(OMe)2', '(MeO)2OP'], '[P](=O)(O[CH3])O[CH3]', "[P](=O)(OC)OC",  0.5),
    Substitution(['P(OH)2'], '[P](O)O', "[P](O)O",  0.5),
    Substitution(['OPO(OEt)2'], '[P](O)(=O)(O[CH2][CH3])O[CH2][CH3]', "[P](O)(=O)(OCC)OCC",  0.5),
    Substitution(['P(OPh)2', '(PhO)2P'], '[P]([O][cH0]1[cH][cH][cH1][cH][cH]1)[O][cH0]1[cH][cH][cH1][cH][cH]1', "[P]([O]c1ccccc1)[O]c1ccccc1", 0.5),
    
    # S-substituted abbreviation
    Substitution(['SMe', 'MeS'], '[SH0;D2][CH3;D1]', "[S]C", 0.3),
    Substitution(['SEt', 'EtS'], '[SH0;D2][CH2;D2][CH3;D1]', "[S]CC", 0.3),
    Substitution(['NfO'], 'S(=O)(=O)[C](F)(F)[C](F)(F)[C](F)(F)[C](F)(F)F', "S(=O)(=O)[C](F)(F)[C](F)(F)F", 0.5),
    Substitution(['SPh', 'PhS'], '[S][cH0]1[cH][cH][cH1][cH][cH]1', "[S]c1ccccc1", 0.5),
    Substitution(['SAda'], '[S][C]12[CH2][CH]3[CH2][CH]([CH2]2)[CH2][CH]([CH2]3)[CH2]1', "[S]C12CC3CC(C2)CC(C3)C1", 0.5),
    Substitution(['STol', 'TolS'], '[S][cH0]1[cH][cH][cH0]([CH3])[cH][cH]1', "[S]c1ccc(C)cc1", 0.5),
    Substitution(['SF3'], '[S](F)(F)F', "[S](F)(F)F", 0.5),
    Substitution(['SCH3'], '[S][CH3]', "[S]C", 0.5),
    Substitution(['SOCH3'], '[S](=O)[CH3]', "[S](=O)C", 0.5),
    Substitution(['SO3Ph', 'PhO3S'], 'S(=O)(=O)O[cH0]1[cH][cH][cH1][cH][cH]1', "[S](=O)(=O)Oc1ccccc1", 0.4),
    Substitution(['SOPh', 'PhOS'], 'S(=O)[cH0]1[cH][cH][cH1][cH][cH]1', "[S](=O)(=O)Oc1ccccc1", 0.4),
    Substitution(['STrt', 'TrtS'], '[SH2;D0]C([cH0]1[cH][cH][cH][cH][cH]1)([cH0]1[cH][cH][cH][cH][cH]1)([cH0]1[cH][cH][cH][cH][cH]1)', "[S]C(c1ccccc1)(c1ccccc1)(c1ccccc1)", 0.5),
    Substitution(['SOtBu'], 'S(=O)[C]([CH3])([CH3])[CH3]', "S(=O)C(C)(C)C", 0.5),
    Substitution(['StBu', 'SBu'], '[S][CH0]([CH3])([CH3])[CH3]', "[S]C(C)(C)C", 0.5),
    
    # Alkyl chains
    Substitution(['C4H9-n', 'nC4H9'], '[CH2][CH2][CH2][CH2][CH3]', "[CH2]CCC", 0.3),
    Substitution(['nC9H19', 'n-C9H19' 'C9H19-n', 'C9H19'], '[CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH3]', "[CH2]CCCCCCCC", 0.3),
    Substitution(['C13H27', 'nC13H27', 'C13H27-n'], '[CH2;D2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH3]', "[CH2]CCCCCCCCCCCC", 0.5),
    Substitution(['C5H11', 'nC5H11', 'C5H11-n'], '[CH2;D2][CH2][CH2][CH2][CH3]', "[CH2]CCCC", 0.5),
    Substitution(['CH(SEt)2', '(SEt)2HC'], '[CH]([SH0;D2][CH2;D2][CH3;D1])[SH0;D2][CH2;D2][CH3;D1]', "[CH]([S]CC)[S]CC", 0.3),
    Substitution(['Me'], '[CH3;D1]', "[CH3]", 0.1),
    Substitution(['Et', 'C2H5'], '[CH2;D2][CH3]', "[CH2]C", 0.3),
    Substitution(['Pr', 'nPr', 'n-Pr'], '[CH2;D2][CH2;D2][CH3]', "[CH2]CC", 0.3),
    Substitution(['Bu', 'nBu', 'n-Bu'], '[CH2;D2][CH2;D2][CH2;D2][CH3]', "[CH2]CCC", 0.3),
    Substitution(['(CH2)14CH3'], '[CH2;D2][CH2;D2][CH2;D2][CH2;D2][CH2;D2][CH2;D2][CH2;D2][CH2;D2][CH2;D2][CH2;D2][CH2;D2][CH2;D2][CH2;D2][CH2;D2][CH3]', "[CH2]CCCCCCCCCCCCCC", 0.3),
    Substitution(['(CH2)14'], '[CH2;D2][CH2;D2][CH2;D2][CH2;D2][CH2;D2][CH2;D2][CH2;D2][CH2;D2][CH2;D2][CH2;D2][CH2;D2][CH2;D2][CH2;D2][CH2;D2]', "[CH2]CCCCCCCCCCCC[CH2]", 0.3),
    
    # Branched
    Substitution(['iPr', 'i-Pr'], '[CH1;D3]([CH3])[CH3]', "[CH1](C)C", 0.2),
    Substitution(['tBu', 't-Bu', 'C(CH3)3'], '[CH0]([CH3])([CH3])[CH3]', "[C](C)(C)C", 0.3),
    Substitution(['iBu', 'i-Bu'], '[CH2;D2][CH1;D3]([CH3])[CH3]', "[CH2]C(C)C", 0.2),
    Substitution(['sBu', 's-Bu'], '[CH1;D3]([CH3])[CH2;D2][CH3]', "[CH](C)CC", 0.2),

    # Other shorthands (MIGHT NOT WANT ALL OF THESE)
    Substitution(['COPh', 'PhOC'], 'C(=O)[cH0]1[cH][cH][cH1][cH][cH]1', "[C](=O)c1ccccc1", 0.4),
    Substitution(['ClF2C', 'CF2Cl'], 'Cl(F)(F)[CH0]', "[C](F)(F)Cl", 0.5),
    Substitution(['CHCl2'], 'Cl(Cl)[CH1]', "[CH](Cl)Cl", 0.5),
    Substitution(['CF2CF3', 'CF3CF3'], '[C](F)(F)[C](F)(F)F', "[C](F)(F)[C](F)(F)F", 0.5),
    Substitution(['(CF2)3CF3'], '[C](F)(F)[C](F)(F)[C](F)(F)[C](F)(F)F', "[C](F)(F)[C](F)(F)[C](F)(F)[C](F)(F)F", 0.5),
    Substitution(['SO3', 'O3S'], 'S(=O)(=O)[O-]', "[S](=O)(=O)[O-]", 0.4),
    Substitution(['OSO3', 'O3SO'], '[O]S(=O)(=O)[O-]', "[O]S(=O)(=O)[O-]", 0.4),
    Substitution(['SO3H', 'HO3S'], 'S(=O)(=O)[OH]', "[S](=O)(=O)O", 0.4),
    Substitution(['H2SO4'], 'OS(=O)(=O)O', "OS(=O)(=O)O", 0.4),
    Substitution(['SO2', 'O2S'], 'S(=O)(=O)', "S(=O)(=O)", 0.4),
    Substitution(['ClOC', 'COCl'], '[CH0](=O)Cl', "[C](=O)Cl", 0.4),
    Substitution(['FOC', 'COF'], '[CH0](=O)F', "[C](=O)F", 0.4),
    Substitution(['CONH2', 'H2NOC', 'H2N(O)C', 'NH2(O)C'], 'C(=O)[NH2]', "[C](=O)N", 0.5),
    Substitution(['CONHMe', 'MeHNOC'], 'C(=O)[NH]C', "[C](=O)N", 0.5),
    Substitution(['SO2NH2'], 'S(=O)(=O)[NH2]', "[S](=O)(=O)N", 0.5),
    Substitution(['SO2Cl'], 'S(=O)(=O)Cl', "[S](=O)(=O)Cl", 0.5),
    Substitution(['CONHtBu', 'tBuHNOC'], '[C](=O)[NH][CH0]([CH3])([CH3])[CH3]', "[C](=O)[NH]C(C)(C)C",  0.5),
    Substitution(['CH(OBn)2', '(OBn)2HC'], '[CH]([OH0;D2][CH2;D2][cH0]1[cH][cH][cH][cH][cH]1)[OH0;D2][CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[CH](OCc1ccccc1)OCc1ccccc1", 0.5),
    Substitution(['MeO2CO', 'OCO2Me', 'MeOOCO', 'OCOOMe'], '[O]C(=O)[OH0;D2][CH3;D1]', "[O]C(=O)OC", 0.5),
    Substitution(['NOH'], '[N][OH]', "[N]O", 0.5),
    Substitution(['NCS', 'SCN', 'NCs'], '[N:D1]=C=S', "[N]=C=S", 0.5),
    Substitution(['NCO', 'OCN'], '[N:D1]=C=O', "[N]=C=O", 0.5),

    # special elements
    Substitution(['SePh', 'PhSe'], '[Se][cH0]1[cH][cH][cH][cH][cH0]1', "[Se]c1ccccc1",  0.5),
    Substitution(['SeMe', 'MeSe'], '[Se][CH3]', "[Se]C",  0.5),
    Substitution(['Sn(Bu)3', 'Bu3Sn', 'SnBu3'], '[Sn]([CH2][CH2][CH2][CH3])([CH2][CH2][CH2][CH3])[CH2][CH2][CH2][CH3]', "[Sn](CCCC)(CCCC)CCCC", 0.5),
    Substitution(['Me3Sn'], '[Sn]([CH3])([CH3])[CH3]', "[Sn](C)(C)C", 0.5),
    Substitution(['Bpin', 'pinB', 'BPin', 'PinB', 'BPIn'], '[BH0]1O[CH0]([CH3])([CH3])[CH0]([CH3])(O1)[CH3]', "[B]1OC(C)(C)C(C)(O1)C",  0.5),
    Substitution(['B(OH)2', '(HO)2B'], '[B]([OH])[OH]', "[B](O)O",  0.5),
    Substitution(['B(OiPr)2', '(OiPr)2B'], '[B]([O][CH1]([CH3])[CH3])[O][CH1]([CH3])[CH3]', "[B](OC(C)(C))OC(C)(C)",  0.5),
    Substitution(['BSia2', 'Sia2B'], '[B]([CH2][CH]([CH3])[CH2][CH3])[CH2][CH]([CH3])[CH2][CH3]', "[B](CC(C)CC)CC(C)CC",  0.5),
    Substitution(['(AcO)3Pb', 'Pb(OAc)3'], '[Pb]([OH0]C(=O)[CH3])([OH0]C(=O)[CH3])[OH0]C(=O)[CH3]', "[Pb]([O]C(=O)C)([O]C(=O)C)[O]C(=O)C",  0.5),
    Substitution(['D3CO', 'COD3'], '[OH0;D2][C]([2H])([2H])[2H]', "[O]C([2H])([2H])[2H]", 0.3),

    # Metal Ion
    Substitution(['NK'], '[K][NH0]', "[N][K]",  0.5),
    Substitution(['NHK'], '[K][NH]', "[NH][K]",  0.5),
    Substitution(['NaO3S', 'SO3Na'], '[Na][OH0][SH0](=O)(=O)', "[S](=O)(=O)O[Na]",  0.5),
    Substitution(['ZnBr'], '[Zn]Br', "[Zn]Br",  0.5),  #ZnBr
    Substitution(['MgBr'], '[Mg]Br', "[Mg]Br",  0.5),  #MgBr
    Substitution(['OMgBr', 'BrMgO'], '[O][Mg]Br', "[O][Mg]Br",  0.5),  #MgBr
    Substitution(['MgCl'], '[Mg]Cl', "[Mg]Cl",  0.5),  #MgCl
    Substitution(['LiO', 'OLi'], '[Li][OH0]', "O[Li]",  0.5),
    Substitution(['SO2Na', 'NaO2S'], 'S(=O)(=O)[Na]', "[S](=O)(=O)[Na]", 0.5),
    Substitution(['NaO', 'ONa'], '[Na][OH0]', "O[Na]",  0.5),
    Substitution(['KO', 'OK'], '[K][OH0]', "O[K]",  0.5),
    Substitution(['SK', 'KS'], '[K][SH0]', "S[K]",  0.5),
    Substitution(['CO2K'], 'C(=O)[OH0;D2][K]', "[C](=O)O[K]", 0.5),
    Substitution(['COOLi', 'CO2Li'], 'C(=O)[OH0;D2][Li]', "[C](=O)O[Li]", 0.5),
    Substitution(['DIPP', 'PPID'], '[P]([O][CH1]([CH3])[CH3])[O][CH1]([CH3])[CH3]', "[P](OC(C)(C))OC(C)(C)",  0.5),

    Substitution(['AsOHTl'], '[CH2;D2]', "[CH2]", 0.0),
    Substitution(['H2PO'], '[CH2;D2]', "[CH2]", 0.0),
    Substitution(['Pd2'], '[CH2;D2]', "[CH2]", 0.0)  


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
