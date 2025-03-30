import os
import ssb64_train as ssb64
from abel_v0 import AbelV0
from abel import Abel
from kane import Kane


kane_list=["kane_lvl_0","kane_lvl_1","kane_lvl_3", "kane_25_from_kane_14_from_kane_10_from_kane_baseline_3","kane_9_from_kane_baseline_1","kane_21_from_kane_17_from_kane_13_from_kane_11_from_kane_baseline_5","kane_24_from_kane_21_from_kane_17_from_kane_13_from_kane_11_from_kane_baseline_5"]
NUM_EPS=100

def main():

    # Test AbelV0 to have a baseline 
    print("Testing AbelV0")

    abelv0=AbelV0("abelv0",player_num=2)
    if os.path.exists("final_vs/abelv0-ai1_results.csv"):
        print("Skip abelv0 ai1 validation")
    else:
        ssb64.test_ai(abelv0,1,num_eps=NUM_EPS)
    if os.path.exists("final_vs/abelv0-ai3_results.csv"):
        print("Skip abelv0 ai3 validation")
    else:
        ssb64.test_ai(abelv0,3,num_eps=NUM_EPS)

    print("Testing Abel")
    # Test abel vs the ai level 0, 1 and 3
    abel=Abel("abel",player_num=1)
    abelv0Opp=AbelV0("abelv0",player_num=2)
    if os.path.exists("final_vs/abel-abelv0_results.csv"):
        print("Skip abel abelv0 validation")
    else:
        ssb64.run_vs(abel,abelv0Opp,num_eps=NUM_EPS)
    if os.path.exists("final_vs/abel-ai1_results.csv"):
        print("Skip abel ai1 validation")
    else:
        ssb64.test_ai(abel,1,num_eps=NUM_EPS)
    if os.path.exists("final_vs/abel-ai3_results.csv"):
        print("Skip abel ai3 validation")
    else:
        ssb64.test_ai(abel,3,num_eps=NUM_EPS)
    abelOpp=Abel("abel",player_num=2)
    for kane_name in kane_list:
        # Test each Kane policy (curriculum learning lvl 0, 1 and 3, and top 4 League Kanes) 
        # against the ai lvl 0,1,3 and the Abel policy
        print("Testing kane ",kane_name)
        kane=Kane(kane_name,kane_name,player_num=1)
        if os.path.exists(f"final_vs/{kane_name}-abelv0_results.csv"):
            print(f"Skip {kane_name} abelv0 validation")
        else:
            ssb64.run_vs(kane,abelv0Opp,num_eps=NUM_EPS)
        if os.path.exists(f"final_vs/{kane_name}-ai1_results.csv"):
            print(f"Skip {kane_name} ai1 validation")
        else:
            ssb64.test_ai(kane,1,num_eps=NUM_EPS)
        if os.path.exists(f"final_vs/{kane_name}-ai3_results.csv"):
            print(f"Skip {kane_name} ai3 validation")
        else:
            ssb64.test_ai(kane,3,num_eps=NUM_EPS)
        if os.path.exists(f"final_vs/{kane_name}-abel_results.csv"):
            print(f"Skip {kane_name} abel validation")
        else:
            ssb64.run_vs(kane,abelOpp,num_eps=NUM_EPS)
    print("DONE DONE DONE DONE")
if __name__ == '__main__':
    main()
