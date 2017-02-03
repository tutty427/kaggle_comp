from numpy import *


defaultW = array([0,0,0])
simpleX = array([[-2,1,-1],[-1,1,-1],[-1,2,-1],[1,1,1],[2,1,1],[2,0.5,1]])
trainTime = 1000;


def sign(result):
    if result > 0: return 1;
    else: return -1;

def SLAModel(w,x):
    Y = dot(w,transpose(append(x[0:2],1)));
    return sign(Y)

def renewWeight(w,x):
    tmp = append(x[0:2],1)*x[2];
    return w+tmp;


def train(w,trainX):
    wf = w
    isstop = False;
    for i in range(trainTime):
        for j in range(simpleX.shape[0]):
            #SLA model
            tmpValue = SLAModel(wf,simpleX[j])
            #compare with the target result
            if tmpValue == simpleX[j][2]:
                if j == simpleX.shape[0]-1:
                    isstop = True
            else:
                wf = renewWeight(wf,simpleX[j])
                print ''.join([str(i),' : ',str(wf)])
                break;
        if isstop:
             break;

    return ''.join(['final : ',str(wf)]);



print train(defaultW,simpleX);