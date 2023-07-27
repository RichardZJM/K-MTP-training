def QE_OUTPUT():
    import sys
    import os
    import csv
    import numpy as np
    import numpy
    import shutil
    import glob
    import math
    import subprocess

    qe_input_dir ='./'
    mlip_conf = open("train.cfg", "w")
    Ry = 13.605693012183622
    Ang = 0.5291772105638411
    F_conversion = Ry/Ang
    S_conversion = Ry/(Ang**3)
    Total = 0

#    files = glob.glob(r'./*')
#    for items in files:
       # os.remove(items)
    # Unity for conversion to eV or angstrom
    for filename in os.listdir(qe_input_dir):
        print(filename)
        if filename.endswith('.out'):
            Total = Total + 1
            with open(os.path.join(qe_input_dir, filename), 'r') as txtfile:
                LINES = txtfile.readlines()
                #print(lines)
                E = []
                Positions = []
                Prim_vector = []
                CELL = []
                lattice = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
                lines = []
 
                index = 0
                for line in LINES:
                    line = line.strip()
                    if line:
                            lines.append(line)
                    if line == 'JOB DONE.':
                        lines.append(line)
                        break
                l_file = len(lines)
                l_job_done = l_file - 2
                
                deadfile = False
                #print(lines[l_file - 2])
                for line in lines:
                    if ('convergence NOT achieved after ' in line):
                        sys.stderr.write('Error: Check QE outputs, job not converged found\n')
                        print('The file name is', filename)
                        exit(1)
                #elif (lines[l_job_done] == 'JOB DONE.'):
              
                if (lines[l_job_done] != 'JOB DONE.'):
                    sys.stderr.write('Warning: Check QE outputs, job not done found. Might be due to local bug. Skipping...\n')
                    print('The file name is', filename)
                    os.remove(filename)
                    deadfile = True
                    continue
                else:
                    index = l_file - 2
                    
                bravais_lattice_index=[j for j in lines if 'bravais-lattice index' in j]
                number_of_atoms_cell=[j for j in lines if 'number of atoms/cell' in j]
                number_of_atomic_types=[j for j in lines if 'number of atomic types' in j]
                lattice_parameter=[j for j in lines if 'lattice parameter' in j]
                Cell_dim = [j for j in lines if 'celldm(1)' in j]
                Nat=int(number_of_atoms_cell[0].split( )[4])
                print(Nat)
                Forces_F = np.zeros((Nat, 3))
                Stress_S = np.zeros((3, 3))
                Positions_P = np.zeros((Nat, 4))
                Energy = 0 
                pattern = 0
                if 'End final coordinates' not in lines:
                    pattern = pattern + 0
                elif 'End final coordinates' in lines and 'Final scf calculation at the relaxed structure.' in lines:
                    pattern = pattern + 1
                else: 
                    pattern =pattern + 2

                #DATA EXTRACTION FROM SCF
                if (pattern==0):
                    for j in range(0, index):
                        if (lines[j].strip()).startswith('crystal axes:') is True:
                            #index_vec = lines.index(lines[j])
                            index_vec = j
                            Cell_cof_scf = float(Cell_dim[0].split()[1])
                            #print(Cell_cof_scf)
                            for s in range(0,3):
                                Prim_vector.append(lines[index_vec + s +1].split())
                                for p in range(0,3):
                                    lattice[s][p] = Ang*Cell_cof_scf*float(Prim_vector[s][p + 3])
                                    #print(lattice)

                        if (lines[j].strip()).startswith('site n.     atom                  positions') is True:
                            #index_P_SCF = lines.index(lines[j])
                            index_P_SCF = j
                            p_scf_ini = index_P_SCF + 1
                            p_scf_fin = index_P_SCF + Nat + 1
                            P_conversion = Ang * Cell_cof_scf
                            for k in range(p_scf_ini,p_scf_fin):
                                Positions = []
                                if (lines[k].split()[1] == 'Na'):
                                    Positions.append(0)
                                else:
                                    Positions.append(1)
                                for i in range(0,3):
                                    #print(lines[k])
                                    Positions.append(P_conversion*float(lines[k].split()[i+6]))
                                convert_P = numpy.array(Positions)
                                P_q = k - p_scf_ini
                                for q in range(0,4):
                                    Positions_P[P_q][q] = convert_P[q]

                        if (lines[j].strip()).startswith('!    total energy') is True:
                            E.append(lines[j])
                            Energy = Ry*float(E[0].split()[4])
                            #print(Energy)
                        if (lines[j].strip()).startswith('Forces acting on atoms') is True:
                            #index_F = lines.index(lines[j])
                            index_F = j
                            f_ini = index_F + 1
                            f_fin = index_F + Nat + 1
                            for i in range(f_ini, f_fin):
                                Forces=[]
                                for j in range(0, 3):
                                    Forces.append(lines[i].split()[j+6])
                                convert_F = numpy.array(Forces)
                                F_k = i - f_ini
                                for l in range(0,3):
                                    Forces_F[F_k][l] = F_conversion*float(convert_F[l])
                        if (lines[j].strip()).startswith('total   stress') is True:
                            #index_S = lines.index(lines[j])
                            index_S = j
                            S_ini = index_S + 1
                            S_fin = index_S + 3 + 1
                            for i in range(S_ini, S_fin):
                                Stress=[]
                                Stress.append(lines[i].split())
                                convert_S = numpy.array(Stress)
                                S_j = i - S_ini
                                for j in range(0,3):
                                    Stress_S[S_j][j] = S_conversion*float(convert_S[0][j])
                        #print(Stress_S)

                # #DATA EXTRACTION FROM VC-RELAX

                # if (pattern==1):
                #     #print(lines.index('Final scf calculation at the relaxed structure.'))
                #     #print(l_file - 2)
                    
                #     index_pattern = lines.index('Final scf calculation at the relaxed structure.')
                #     index_end = l_file - 2
                #     #lines = lines[index_pattern:index_end -1]

                #     for j in range(index_pattern, index_end):
                #         if (lines[j].strip()).startswith('crystal axes:') is True:
                #             #index_vec = lines.index(lines[j])
                #             index_vec = j
                #             Cell_cof_scf = float(Cell_dim[0].split()[1])
                #             #print(Cell_cof_scf)
                #             for s in range(0,3):
                #                 Prim_vector.append(lines[index_vec + s +1].split())
                #                 for p in range(0,3):
                #                     lattice[s][p] = Ang*Cell_cof_scf*float(Prim_vector[s][p + 3])
                #                     #print(lattice)

                #         if (lines[j].strip()).startswith('site n.     atom                  positions') is True:
                #             #index_P_SCF = lines.index(lines[j])
                #             index_P_SCF = j
                #             p_scf_ini = index_P_SCF + 1
                #             p_scf_fin = index_P_SCF + Nat + 1
                #             P_conversion = Ang * Cell_cof_scf
                #             for k in range(p_scf_ini,p_scf_fin):
                #                 Positions = []
                #                 if (lines[k].split()[1] == 'Na'):
                #                     Positions.append(0)
                #                 else:
                #                     Positions.append(1)
                #                 for i in range(0,3):
                #                     #print(lines[k])
                #                     Positions.append(P_conversion*float(lines[k].split()[i+6]))
                #                 convert_P = numpy.array(Positions)
                #                 P_q = k - p_scf_ini
                #                 for q in range(0,4):
                #                     Positions_P[P_q][q] = convert_P[q]

                #         if (lines[j].strip()).startswith('!    total energy') is True:
                #             E.append(lines[j])
                #             Energy = Ry*float(E[0].split()[4])
                #             #print(Energy)
                #         if (lines[j].strip()).startswith('Forces acting on atoms') is True:
                #             #index_F = lines.index(lines[j])
                #             index_F = j
                #             f_ini = index_F + 1
                #             f_fin = index_F + Nat + 1
                #             for i in range(f_ini, f_fin):
                #                 Forces=[]
                #                 for j in range(0, 3):
                #                     Forces.append(lines[i].split()[j+6])
                #                 convert_F = numpy.array(Forces)
                #                 F_k = i - f_ini
                #                 for l in range(0,3):
                #                     Forces_F[F_k][l] = F_conversion*float(convert_F[l])
                #         if (lines[j].strip()).startswith('total   stress') is True:
                #             #index_S = lines.index(lines[j])
                #             index_S = j
                #             S_ini = index_S + 1
                #             S_fin = index_S + 3 + 1
                #             for i in range(S_ini, S_fin):
                #                 Stress=[]
                #                 Stress.append(lines[i].split())
                #                 convert_S = numpy.array(Stress)
                #                 S_j = i - S_ini
                #                 for j in range(0,3):
                #                     Stress_S[S_j][j] = S_conversion*float(convert_S[0][j])

                # #DATA EXTRACTION FOR RELAX

                # else:
                #     for j in range(0, index):
                #         if (lines[j].strip()).startswith('End final coordinates') is True:
                #             #index_P = lines.index(lines[j])
                #             index_P = j
                #             #print(index_P)
                #             p_ini = index_P - Nat
                #             p_fin = index_P
                #             cell_ini = p_ini - 3 -1 -1
                #             cell_fin = p_ini - 1

                #             for i in range(p_ini, p_fin):
                #                 Positions = []
                #                 if (lines[i].split()[0] == 'Na'):
                #                     Positions.append(0)
                #                 else:
                #                     Positions.append(1)
                                    
                #                 for k in range(0, 3):
                #                     Positions.append(lines[i].split()[k + 1])
                #                 convert_P = numpy.array(Positions)
                #                 P_q = i - p_ini
                #                 for q in range(0, 4):
                #                     Positions_P[P_q][q] = convert_P[q]
                #             index_ll=[]
                #             for ll in range(index_P, 0, -1):
                #                 if (lines[ll].strip()).startswith('!    total energy') is True:
                #                     index_ll.append(ll)
                #             E_index = index_ll[0]
                #             Energy = Ry*float(lines[E_index].split()[4])
                #             for ll_f in range(E_index, index_P):
                #                 if (lines[ll_f].strip()).startswith('Forces acting on atoms') is True:
                #                     #index_F = ll_f
                #                     F_ini = ll_f + 1
                #                     F_fin = ll_f + Nat + 1
                #                     for ff in range(F_ini, F_fin):
                #                         Forces = []
                #                         for j in range(0, 3):
                #                             Forces.append(lines[ff].split()[j + 6])
                #                         convert_F = numpy.array(Forces)
                #                         F_k = ff - F_ini
                #                         for l in range(0, 3):
                #                             Forces_F[F_k][l] = F_conversion*float(convert_F[l])
                #             for ll_s in range(E_index, index_P):
                #                 if (lines[ll_s].strip()).startswith('total   stress') is True:
                #                     S_ini = ll_s + 1
                #                     S_fin = ll_s + 3 + 1
                #                     for i in range(S_ini, S_fin):
                #                         Stress = []
                #                         Stress.append(lines[i].split())
                #                         convert_S = numpy.array(Stress)
                #                         S_j = i - S_ini
                #                         for j in range(0, 3):
                #                             Stress_S[S_j][j] = S_conversion*float(convert_S[0][j])


                #             if 'CELL_PARAMETERS' in lines[cell_ini]:
                #                 for i in range(cell_ini, cell_fin):
                #                     CELL.append(lines[i].split())
                #                 Cell_cof = float(CELL[0][2].strip('()'))
                #                 for p in range(0, 3):
                #                     for q in range(0, 3):
                #                         lattice[p][q] = Ang*Cell_cof * float(CELL[p + 1][q])
                #             else:
                #             #if 'CELL_PARAMETERS' not in lines[cell_ini]:
                #                 Cell_cof_scf = float(Cell_dim[0].split()[1])
                #                 for j in range(0, index):
                #                     if (lines[j].strip()).startswith('crystal axes:') is True:
                #                         index_vec = lines.index(lines[j])
                #                         for s in range(0, 3):
                #                              Prim_vector.append(lines[index_vec + s + 1].split())
                #                              for p in range(0, 3):
                #                                  lattice[s][p] = Ang*Cell_cof_scf*float(Prim_vector[s][p + 3])

    #cell_volume calculation via scalar product
            txtfile.close()

            A2vA3_1 = lattice[1][1] * lattice[2][2] - lattice[1][2] * lattice[2][1]
            A2vA3_2 = lattice[1][2] * lattice[2][0] - lattice[1][0] * lattice[2][2]
            A2vA3_3 = lattice[1][0] * lattice[2][1] - lattice[1][1] * lattice[2][0]
            print(A2vA3_1)
            V = lattice[0][0] * A2vA3_1 + lattice[0][1] * A2vA3_2 + lattice[0][2] * A2vA3_3
            print(V)
            cfg_size = Nat
            #mlip file cfg         
            mlip_conf.write("BEGIN_CFG\n")
            mlip_conf.write(" Size\n")
            mlip_conf.write("    " + str(cfg_size) + "\n")
            mlip_conf.write(" Supercell\n")
            mlip_conf.write("    " + str(lattice[0][0]) + "     " + str(lattice[0][1]) + "      " + str(lattice[0][2]) + " \n")
            mlip_conf.write("    " + str(lattice[1][0]) + "     " + str(lattice[1][1]) + "      " + str(lattice[1][2]) + " \n")
            mlip_conf.write("    " + str(lattice[2][0]) + "     " + str(lattice[2][1]) + "      " + str(lattice[2][2]) + " \n")
                
            mlip_conf.write(" AtomData:  id type    cartes_x    cartes_y    cartes_z    fx    fy    fz\n")
             
            for i in range(0, cfg_size):
                    typ = int(Positions_P[i][0])
                    pos_x = Positions_P[i][1]
                    pos_y = Positions_P[i][2]
                    pos_z = Positions_P[i][3]
                    f_x = Forces_F[i][0]
                    f_y = Forces_F[i][1]
                    f_z = Forces_F[i][2]
                    mlip_conf.write(
                        "    " + str(i+1) + "    " + str(0) + "   " + str(pos_x) + "    " + str(pos_y) + "    " + str(
                            pos_z) + "    " + str(f_x) + "    " + str(f_y) + "    " + str(f_z) + "\n")
            xx = V*Stress_S[0][0]
            yy = V*Stress_S[1][1]
            zz = V*Stress_S[2][2]
            yz = V*Stress_S[1][2]
            xz = V*Stress_S[0][2]
            xy = V*Stress_S[0][1]
            mlip_conf.write(" Energy\n")
            mlip_conf.write("    " + str(Energy) + "\n")
            mlip_conf.write(" PlusStress:  xx          yy          zz          yz          xz          xy\n")
            mlip_conf.write(
                    "    " + str(xx) + "   " + str(yy) + "    " + str(zz) + "    " + str(
                        yz) + "    " + str(xz) + "    " + str(xy) + "\n")
            mlip_conf.write(" Feature EFS_by Qe\n")
            mlip_conf.write("END_CFG\n")
            mlip_conf.write("\n")
        
    mlip_conf.close()

                        #print(Positions_P)
                        #print(Cell_dim)
                        #print(lattice)
                        #print(Energy)
                        #print(Forces_F)
                        #print(Stress_S)
                        #print(Energy)
    
    print("Total number of processed configurations = " +  str(Total))

QE_OUTPUT()
