from random import randint 

# @param trials: Number of flips to perform
# @param maxdif: Maximum allowed difference of heads and tails
# @return [res, nH, nT, prices]:
# @return res: list of length trials of H and T flips
# @return nH: number of heads at the end of the trials
# @return nT: number of tails at the end of the trials
# @return prices: list of expected values for final payout
def generate(trials, maxdif):
  # Boolean flag to check validity of outputted HT sequence
  flag = False
  while not flag:
    # Always start with a heads
    res = ['H'] + [''] * (trials - 1)
    nH, nT = 1, 0
    for i in range(1, trials):
      # If the difference is equal to or exceeds the maxdif, automatically return tails
      if nH - nT >= maxdif:
        res[i] += 'T'
        nT += 1
      else:
        # Fairly flip H or T
        if randint(0,1) == 0:
          res[i] += 'H'
          nH += 1
        else:
          res[i] += 'T'
          nT += 1
          # If nT > nH, regenerate the entire sequence (failed too early)
          if nT > nH:
            flag = False
            break
      
      # Successfully generated sequence
      if i == trials - 1:
        flag = True
  
  # Expected value of payout. Determined by Monte Carlo simulation
  prices = [[50.107, 124.766, 250.33, 499.742, 998.344, 1199.368],
            [62.498, 150.82, 312.149, 625.213, 851.114, 1097.492],
            [75.133, 187.3555, 388.251, 581.1985, 865.342, 974.582],
            [94.506, 231.58, 383.009, 624.29, 776.8965, 919.013],
            [115.482, 240.064, 426.6, 579.284, 772.713, 847.3665],
            [120.275, 272.339, 411.238, 600.536, 717.167, 811.0235],
            [133.804, 265.085, 437.494, 564.4615, 705.9525, 763.584],
            [132.939, 285.11, 416.527, 567.9795, 661.426, 732.046],
            [142.544, 275.1505, 425.1415, 537.35, 649.103, 697.0025],
            [135.6485, 286.8195, 405.9365, 540.0255, 615.0205, 676.295],
            [143.821, 271.0815, 415.3015, 509.7255, 604.746, 648.222],
            [136.652, 279.1945, 388.863, 508.557, 579.3795, 626.9335],
            [137.011, 266.7775, 394.874, 484.9755, 569.1175, 605.8625],
            [131.224, 265.765, 376.315, 482.5495, 546.9405, 587.9725],
            [132.668, 252.805, 375.777, 459.541, 533.44, 563.74],
            [127.052, 254.548, 359.6705, 452.7275, 515.6845, 549.631],
            [126.871, 240.5185, 354.1445, 435.0095, 502.159, 528.2515],
            [121.1125, 240.9865, 336.7205, 427.139, 485.413, 514.7255],
            [119.254, 227.717, 332.6665, 406.349, 471.325, 500.7105],
            [115.859, 230.0995, 319.3775, 403.679, 455.175, 486.63]]


  return [res, nH, nT, prices]


def simulation(n):
  run = generate(n,5)
  # Guarantee the ending of the game by forcing a sequence of tails
  HT_list = run[0] + ['T'] * (run[1] - run[2] + 1)
  nH, nT = 0, 0
  for index in range(0,len(HT_list)):
    print(HT_list[index])
    if HT_list[index] == 'H':
      nH += 1
    else:
      nT += 1

    if nT > nH:
      print('Game ended! Remaining units are worth 0.')
      break
    
    base_price = run[3][20 - (nH + nT)][nH-nT]
    # A noise multiplier from 0.98 to 1.02
    mult = randint(98,102) / 100

    print('Number of heads: ' + str(nH))
    print('Number of tails: ' + str(nT))
    print('Price of Unit: ' + str(int(mult * base_price)))

    input("Next round?\n")
    print('\n')
  return 0

# Simulate for an unknown number of flips around 20
simulation(randint(14,20))