from scipy.stats import wilcoxon, mannwhitneyu
import numpy as np

files = [
    "instances/low-dimensional/f1_l-d_kp_10_269_results.csv",
    "instances/low-dimensional/f2_l-d_kp_20_878_results.csv",
    "instances/low-dimensional/f3_l-d_kp_4_20_results.csv",
    "instances/low-dimensional/f4_l-d_kp_4_11_results.csv",
    "instances/low-dimensional/f6_l-d_kp_10_60_results.csv",
    "instances/low-dimensional/f7_l-d_kp_7_50_results.csv",
    "instances/low-dimensional/f8_l-d_kp_23_10000_results.csv",
    "instances/low-dimensional/f9_l-d_kp_5_80_results.csv",
    "instances/low-dimensional/f10_l-d_kp_20_879_results.csv",
    "instances/results/knapPI_1_100_1000_1.csv",
    "instances/results/knapPI_1_200_1000_1.csv",
    "instances/results/knapPI_1_500_1000_1.csv",
    "instances/results/knapPI_1_1000_1000_1.csv",
    "instances/results/knapPI_2_200_1000_1.csv",
    "instances/results/knapPI_2_500_1000_1.csv",
    "instances/results/knapPI_2_1000_1000_1.csv",
    "instances/results/knapPI_3_100_1000_1.csv",
    "instances/results/knapPI_3_200_1000_1.csv",
    "instances/results/knapPI_3_500_1000_1.csv",
    "instances/results/knapPI_3_1000_1000_1.csv"
]

strats = [
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"
]

for instancia in files:

    print(instancia)

    archivo = open(instancia, "r")

    lines = archivo.readlines()

    matrix = np.zeros(shape=(len(strats),len(strats)))


    for i in range(len(strats)):
        for j in range(len(strats)):
            if i != j:
                linea_a = lines[i].strip().split(',')
                linea_b = lines[j].strip().split(',')
                dist_a = [int(linea_a[x]) for x in range(1,len(linea_a))]
                dist_b = [int(linea_b[x]) for x in range(1,len(linea_b))]
                #print(f"{i},{j} {wilcoxon(dist_a,dist_b).pvalue}"

                matrix[i,j] = mannwhitneyu(dist_a,dist_b, alternative="greater").pvalue

    #print(matrix)

    output = open(f"tabla_{instancia.split('/')[2]}.txt", 'w')

    output.write("""
    \\begin{table}[h]
    \centering
    \\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|c|}
    \hline
    Impl. & C1 & C2 & C3 & C4 & C5 & C6 & C7 & C8 & C9 & C10\\\\ \hline
    """)

    for i in range(matrix.shape[0]):
        output.write(f"C{i+1} &")
        for j in range(matrix.shape[1]):
            write = True
            if i == j:
                output.write("-")
                write = False
            if write:
                if matrix[i][j] <= 0.05:
                    output.write("{:.1e}".format(matrix[i][j]) + " ")
                else:
                    output.write("> 0.05 ")
            if j != matrix.shape[1]-1:
                output.write("& ")
            else:
                output.write("\\\\")
        if i == matrix.shape[0]-1:
            output.write(" \hline")
        output.write('\n')

    output.write("""
    \end{tabular}
    """)
    output.write("\\caption{Resultados test estadistico configuraciones C1-C10 instancia " +  instancia.split('/')[2].replace('_', '-').replace("-results.csv","") + "}")
    output.write("""
    \label{table:ch}
    \end{table}
    """)

    output.close()