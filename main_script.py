# main_script.py

import prepare_iris 
import train_regression1, train_regression2, train_regression3, train_regression4, train_regression_multiple_outputs, train_regression_regularized
import train_classifier1, train_classifier2, train_classifier3
import eval_regression1, eval_regression2, eval_regression3, eval_regression4
import eval_classifier1, eval_classifier2, eval_classifier3

def main():
    prepare_iris.main()

    train_regression1.main()
    eval_regression1.main()

    train_regression2.main()
    eval_regression2.main()

    train_regression3.main()
    eval_regression3.main()

    train_regression4.main()
    eval_regression4.main()

    train_regression_multiple_outputs.main()

    train_regression_regularized.main()

    train_classifier1.main()
    eval_classifier1.main()

    train_classifier2.main()
    eval_classifier2.main()

    train_classifier3.main()
    eval_classifier3.main()

if __name__ == "__main__":
    main()