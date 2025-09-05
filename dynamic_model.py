# entry point for the dynamic model estimation
from time import perf_counter
import cohorts
import getopt
import sys

def usage(proc):
    print("usage: " + proc +
          "\n\t-s --static: do not calculate emax" +
          "\n\t-m --moments: display moments" +
          "\n\t-d --dump-emax: dump emax matrices into files" +
          "\n\t-c --cohort: cohort. e.g. 1970white" +
          "\n\t-v --verbose")
    exit(0)


def main():
    short_options = "hsvmdc:"
    long_options = ["help", "static", "verbose", "moments", "dump-emax", "cohort="]
    display_moments = False
    verbose = False
    static_model = False
    dump_emax = False
    try:
        opts, args = getopt.getopt(sys.argv[1:], short_options, long_options)
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        usage(sys.argv[0])

    for opt, arg in opts:
        if opt in ("-h", "--Help"):
            usage(sys.argv[0])
        elif opt in ("-m", "--moments"):
            display_moments = True
        elif opt in ("-v", "--verbose"):
            verbose = True
        elif opt in ("-s", "--static"):
            static_model = True
        elif opt in ("-d", "--dump-emax"):
            dump_emax = True
        elif opt in ("-c", "--cohort"):
            cohorts.cohort = arg

    if cohorts.cohort is None or cohorts.cohort == "":
        print("'cohort' is a mandatory parameter")
        usage(sys.argv[0])

    if static_model and dump_emax:
        print("when running a static model emax cannot be dumped")

    if verbose:
        print("configuration: verbose output")
    if display_moments:
        print("configuration: displaying calculated moments")
    else:
        print("configuration: don't display calculated moments")
    if static_model:
        print("configuration: no emax calculation")
    else:
        print("configuration: emax will be calculated")
    if dump_emax:
        print("configuration: emax will be dumped to a file")
    else:
        print("configuration: emax will not be dumped")
    print("configuration: running on "+cohorts.cohort+" cohort")

    # these imports mus tbe done *after* the cohorts global parameter is set
    from calculate_emax import create_married_emax
    from calculate_emax import create_single_w_emax
    from calculate_emax import create_single_h_emax
    from calculate_emax import calculate_emax
    from calculate_emax import dump_married_emax, dump_single_emax
    import forward_simulation as fs

    w_emax = create_married_emax()
    h_emax = create_married_emax()
    w_s_emax = create_single_w_emax()
    h_s_emax = create_single_h_emax()
    if not static_model:
        tic = perf_counter()
        iter_count = calculate_emax(w_emax, h_emax, w_s_emax, h_s_emax, verbose)
        toc = perf_counter()
        print("calculate emax with %d iterations took: %.4f (sec)" % (iter_count, (toc - tic)))
        if dump_emax:
            dump_married_emax("w_emax", w_emax)
            dump_married_emax("h_emax", h_emax)
            dump_single_emax("w_s_emax", w_s_emax)
            dump_single_emax("h_s_emax", h_s_emax)


    value = fs.forward_simulation(w_emax, h_emax, w_s_emax, h_s_emax, verbose, display_moments)
    print(value)


if __name__ == "__main__":
    main()
