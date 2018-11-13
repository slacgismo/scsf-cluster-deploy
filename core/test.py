import math, sys, time
import pp


def isprime(n):
    """Returns True if n is prime and False otherwise"""
    if not isinstance(n, int):
        raise TypeError("argument passed to is_prime is not of 'int' type")
    if n < 2:
        return False
    if n == 2:
        return True
    max = int(math.ceil(math.sqrt(n)))
    i = 2
    while i <= max:
        if n % i == 0:
            return False
        i += 1
    return True

def sum_primes(n):
    """Calculates sum of all primes below given integer n"""
    return sum([x for x in range(2,n) if isprime(x)])


# tuple of all parallel python servers to connect with
ppservers = ()
#ppservers = ("54.219.180.66:35000", "52.53.222.244:35000", "18.144.6.145:35000", "13.57.254.165:35000")
#ppservers = ("10.0.0.1",)

if len(sys.argv) > 1:
    ncpus = int(sys.argv[1])
    # Creates jobserver with ncpus workers
    job_server = pp.Server(ncpus, ppservers=ppservers, secret="this1saj3st")
else:
    # Creates jobserver with automatically detected number of workers
    job_server = pp.Server(ppservers=ppservers, secret="this1saj3st")

print("Starting pp with", job_server.get_ncpus(), "workers")

start_time = time.time()

inputs = (100000, 100100, 100200, 100300, 100400, 100500, 100600, 100700, 100800, 100900, 101000, 101100, 101200, 101300, 101400, 101500, 101600, 101700, 101800, 101900)
#inputs = (100000, 100100, 100200, 100300, 100400, 100500, 100600, 100700)
jobs = [(input, job_server.submit(sum_primes,(input,), (isprime,), ("math",))) for input in inputs]
for input, job in jobs:
    print("Sum of primes below", input, "is", job())

print("Time elapsed: ", time.time() - start_time, "s")
job_server.print_stats()
