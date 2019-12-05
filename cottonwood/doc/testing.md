# Testing Philosophy

I have found that for small single-developer research projects, the best balance between
reliability and the time required to ensure it is not to write unit tests. Instead,
a small set of functional tests that touch most of the code in some way serves as
a canary in our coal mine. As long as all the examples run well, chances are that
we haven't broken anything badly. It doesn't ensure that every function is calculating
exactly what we want it to. In my experience there is always another bug in the code.
However, since our code is for research purposes we can assume that it will be changing
rapidly as we generate and test new ideas. If ever we decide to mature it to production
code, suitable for long-term maintenance by a large team and automatic deployment,
there are many steps we would need to take. Writing unit tests would be one of them. 
