import math


def cosine(v1, v2):
    return 1.0*(sum(i*j for i,j in zip(v1,v2)))/(math.sqrt(sum(i**2 for i in v1)) + math.sqrt(sum(j**2 for j in v2)))


def dice(v1, v2):
    return 2.0*(sum(min(i, j) for i,j in zip(v1,v2)))/(sum(i+j for i,j in zip(v1,v2)))


def euclidean(v1, v2):
    return math.sqrt(sum((i-j)**2 for i,j in zip(v1,v2)))


if __name__ == '__main__':
    w1 = '0.15398091 -0.0021545414 -0.37327617 -0.34287238 0.035410687 0.14822531 0.080427505 0.24715328 0.031548027 -0.18533069 -0.033056807 -0.2711267 0.31974652 -0.28519255 -0.4358613 -0.19339715 -0.05440687 -0.06455012 0.15502062 0.056935538 -0.19751588 -0.11665373 -0.48288935 -0.44492683 -0.046249293 0.068822786 0.12309465 -0.0045951246 -0.29060265 0.011900001 0.41595045 -0.4055516 0.52914304 -0.31329998 -0.12107623 0.49284747 0.10279098 0.14681493 -0.20820186 0.121168785 -0.3422114 0.3104283 -0.43623194 -0.51055294 -0.28645718 0.038605634 -0.17055504 -0.28241864 -0.34884977 0.10331553 0.35430774 -0.36506674 0.28740847 0.35698026 -0.099903926 0.015984237 -0.5844623 0.3266663 -0.3615238 -0.19648184 0.10458129 0.5249436 0.04242836 0.36636776 0.04124446 -0.035221342 -0.27869493 0.20420405 0.30417302 -0.40650967 0.013405177 -0.22420155 -0.020095827 -0.15937634 -0.2728807 -0.021344442 -0.08299765 -0.103912644 -0.2280472 -0.28124183 0.30525675 -0.0303815 -0.061614513 -0.23818514 -0.08186722 -0.08261211 -0.32988477 0.1923242 0.39514634 -0.11284162 0.006924306 -0.02987208 -0.05913543 -0.24720801 0.44477397 0.16653724 0.27130884 -0.08359019 -0.34673256 -0.11968421 -0.4763083 -0.025316983 0.427599 -0.46622065 -0.09237725 -0.1546114 0.060754552 -0.23317245 -0.44196475 0.12865818 0.36736286 -0.37545562 -0.08128668 -0.23139252 0.055630516 0.5255092 -0.017250864 -0.18567666 -0.44220865 0.2444123 -0.14529246 0.3667704 -0.13053027 0.31857696 0.31893623 -0.097047046 -0.24757749 0.23950118 0.20547745 -0.06381271 -0.30830798 0.2130629 -0.61004794 0.021057349 0.3135658 -0.008984547 0.3051218 0.031727787 -0.17309494 -0.072780035 -0.4085408 0.06388071 -0.1348205 -0.5138349 -0.15080197 -0.26964974 -0.21598618 0.0054005054 -0.45561436 0.34358972'
    w2 = '0.2133356 0.09087379 -0.30182606 -0.1683613 0.0066878423 -0.089371935 0.023286806 0.34618202 0.002152507 -0.49690205 0.10896605 -0.020771606 0.07729501 0.10475283 -0.21497911 -0.15755792 0.028300703 0.02850864 0.1300774 -0.119520724 0.07037526 -0.2331096 0.0091026435 -0.36714152 0.100997984 0.21459165 0.027185936 -0.21050099 -0.058564223 0.11099297 0.2894009 -0.17237833 0.20541023 -0.11373386 0.030773716 0.15186197 0.2342943 0.051259536 -0.15863797 -0.13871157 -0.32159266 0.053823743 -0.16098139 -0.021428024 0.14047025 0.18350405 -0.13118795 -0.3355216 -0.016387327 -0.02071883 0.08103126 0.03889923 -0.16142903 0.055748325 -0.2778487 0.08209038 -0.28043023 0.060623787 -0.21007818 0.21369499 -0.17609671 -0.0011169359 0.14921632 0.43842858 -0.054882914 -0.023935787 -0.5193197 -0.12113516 -0.061084956 -0.18488482 0.037003003 -0.11452263 0.080228396 0.055410992 -0.23338261 0.14044724 0.050362386 0.005064889 0.2663917 -0.0005515685 0.18798167 -0.28783333 0.024710549 0.08233477 0.0660183 0.00972663 -0.0042225155 0.26861084 -0.09259802 -0.30166346 -0.029105676 -0.15107611 -0.15442364 -0.38296717 -0.049819577 -0.44287536 -0.031713918 0.06467569 -0.21169496 -0.26721808 -0.09897686 -0.1279315 0.25090563 -0.14607978 -0.110787414 -0.2525955 0.18300699 -0.30470553 -0.3401339 0.06805067 0.13732208 -0.19941008 0.030603318 -0.2087909 -0.036118865 0.19050142 0.06646301 -0.088342026 -0.18781206 -0.009674877 0.13549747 0.15514469 0.091259025 -0.061813213 0.33208498 -0.17994422 -0.16249155 -0.23089164 0.0060687065 0.20154783 -0.28928682 0.021885967 -0.33277142 0.05415737 0.24236527 0.15637735 0.16614012 0.1461109 -0.08684747 -0.23274495 -0.30544567 -0.05596004 -0.08024131 -0.27633533 -0.07999444 -0.23753744 -0.0075851246 0.027667891 0.0023656145 0.28365052'
    w3 = '0.113769926 0.019604547 -0.24869503 -0.100111745 0.075506784 0.005606348 0.07259001 0.039404724 0.06755415 -0.16270357 -0.03325508 -0.016159054 0.11730039 -0.048655886 -0.23823509 -0.035049547 0.12300069 0.02289854 -0.045249444 0.02781466 -0.056045294 -0.031252462 -0.1302638 -0.29011446 0.07643621 0.07439091 0.037426062 -0.042064097 0.012994621 0.003926345 0.13548928 -0.09652149 0.120568365 0.044595703 0.05869846 0.12931721 0.12544227 0.066038646 -0.103029005 -0.041987922 -0.08657714 0.18912242 -0.27556506 -0.16818957 0.024598949 0.17714526 -0.042426568 -0.18511555 -0.13549928 0.08720182 0.22684172 -0.023760263 0.08899769 0.20223609 -0.20811565 0.1261587 -0.28679377 0.067618 -0.14870335 0.024829496 -0.13524248 0.005609741 0.14478531 0.282935 -0.060434263 -0.062539235 -0.27142763 0.122217506 0.17811657 -0.1505798 0.093545936 -0.12237389 0.003997757 0.066677935 -0.29712686 0.07510397 0.028713917 -0.0594072 0.08030708 0.04955119 0.11740117 -0.14062189 -0.097722754 0.05898917 -0.078856245 -0.06576518 -0.12265579 0.16998301 0.10069107 0.030400954 -0.123085596 -0.15077524 -0.05472794 -0.20603487 0.20308436 -0.06221584 0.19927318 0.13372779 -0.09489484 -0.03786437 -0.123570435 -0.13907512 0.06304456 -0.15022047 -0.15122621 -0.19074716 0.08964607 -0.15839201 -0.25329185 0.051392294 0.13641445 -0.12680154 -0.12457099 -0.22355245 -0.002115043 0.21490218 0.09057632 -0.08982976 -0.16891783 0.025815567 -0.01718769 0.1266147 -0.013036167 -0.066680096 0.17708758 -0.22229482 -0.13429104 0.016176362 0.108169094 -0.096593395 -0.17562546 -0.029007792 -0.27257028 0.102755755 0.15283346 -0.07748274 0.19702524 0.16317539 -0.0032851235 0.0262003 -0.22553836 -0.034652665 -0.09624078 -0.2030216 -0.09792553 -0.22584274 0.038254604 -0.028973637 -0.09796903 0.17295863'

    v1 = list(map(float, w1.split()))
    v2 = list(map(float, w2.split()))
    v3 = list(map(float, w3.split()))

    print(cosine(v1, v2))
    print(cosine(v2, v3))

    print(dice(v1, v2))
    print(dice(v2, v3))

    print(euclidean(v1, v2))
    print(euclidean(v2, v3))