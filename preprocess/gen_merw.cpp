#include <bits/stdc++.h>
#include <cmath>
#define N 100050
#define N2 20
using namespace std;

const double eps = 1e-5;

int num_of_walks = 40;
int seq_len = 6;
int o[20];
char dis[N][N];
bool vis[N];
int n, m;

vector<int> E[N]; //表示一个长度为N的数组，每个元素是一个vector<int>类型的向量。这个数组中的每个元素都是一个动态数组，可以存储多个整数。
vector<double> p_merw[N];
vector<int> T[N];
int timestamps[20];
float t[20];
int totalEpochs = 15; //要与co-train里面的epochs数量已有


/*自定义的类，它有三个成员变量 A, B, 和 S，它们都是 vector 类型。
这个类还有一个构造函数和一个名为 init 的方法。*/
class AliasTable{
    public:
        vector<int> A, B;
        vector<double> S;
    public:
        AliasTable () {}
        // init方法用于初始化Alias Table。
        void init(vector<int> &a, vector<double> &p) {
            queue<int> qA, qB;
            queue<double> pA, pB;
            int n = (int)a.size();

            for (int i=0;i<n;i++) p[i] = p[i] * n;
            for (int i=0;i<n;i++)
                if (p[i] > 1.0) {
                    qA.push(a[i]);
                    pA.push(p[i]);
                } else {
                    qB.push(a[i]);
                    pB.push(p[i]);
                }
            while (!qA.empty() && !qB.empty()) {
                int idA = qA.front(); qA.pop();
                double probA = pA.front(); pA.pop();
                int idB = qB.front(); qB.pop();
                double probB = pB.front(); pB.pop();

                A.push_back(idA);
                B.push_back(idB);
                S.push_back(probB);

                double res = probA-(1.0-probB);

                if (abs(res-1.0) < eps) {
                    A.push_back(idA);
                    B.push_back(idA);
                    S.push_back(res);
                    continue;
                }
                // 不符合条件又入队qA、pA
                if (res > 1.0) {
                    qA.push(idA);
                    pA.push(res);
                } else {
                    qB.push(idA);
                    pB.push(res);
                }
            }

            while (!qA.empty()) {
                int idA = qA.front(); qA.pop();
                pA.pop();
                A.push_back(idA);
                B.push_back(idA);
                S.push_back(1.0);
            }

            while (!qB.empty()) {
                int idB = qB.front(); qB.pop();
                pB.pop();
                A.push_back(idB);
                B.push_back(idB);
                S.push_back(1.0);
            }
        }
        // 从Alias表中进行随机抽样，判断返回节点是A[x]还是B[x]
        int roll() {
	        // if ((int)A.size() == 0) {
		    //     cerr << "ERROR:: A.size() == 0 in Alias Table " << (int)A.size() <<endl;
		    //     exit(0);
	        // }

            /*解决TKG 数据集中节点在train/valid/test不存在的问题
            如果7120不存在，则会形成路径[7120, 7120, 7120, 7120, 0, 0, 0, 0]输出
            */
            if ((int)A.size() == 0) {
		        return -1;
	        }
            int x = rand() % ((int)A.size());//生成一个随机整数x，范围是[0, A.size())，即在向量A的索引范围内随机选择一个整数。
            double p = 1.0 * rand() / RAND_MAX;//生成一个随机概率p，范围是[0, 1.0)
            //根据生成的随机概率p和Alias表中的S向量的值，判断返回A[x]还是B[x]。
            //如果p大于S[x]，则返回A[x]，否则返回B[x]
            return p>S[x] ? A[x] : B[x];
        }

}AT[N];

void link(int u, int v, int t, double p)
{
    E[u].push_back(v);
    T[u].push_back(t); //add time
    p_merw[u].push_back(p);
}

// 从节点S开始进行广度优先搜索，计算节点之间的最短距离,存储在二维数组dis中
void bfs(int S)
{
    queue<int> q;
    q.push(S);
    dis[S][S] = 1;
    while (!q.empty())
    {
        int u = q.front();
        q.pop();
        if (dis[S][u] > seq_len)
            return;
        for (int i = 0; i < (int)E[u].size(); i++)
        {
            int v = E[u][i];
            if (dis[S][v] == 0)
            {
                dis[S][v] = dis[S][u] + 1;
                q.push(v);
            }
        }
    }
    return;
}

int findIndex(const std::vector<int> subvector, int value) {
    for (size_t i = 0; i < subvector.size(); ++i) {
        if (subvector[i] == value) {
            return i; // 返回元素在子向量中的索引
        }
    }
    return -1; // 如果没有找到元素，返回-1
}

int main(int argc, char *argv[])
{

    if (argc != 4)
    {
        cerr << "ERROR: Incorrect number of parameters. " << endl;
        return 0;
    }

    stringstream ss1, ss2; // ss1和ss2分别是输入和输出文件名,可以随意修改文件的路径以适应您的运行环境。
    ss1.str("");
    ss1 << "../edge_input/";  //这里报错一整天，只能用绝对路径
    ss1 << argv[1];
    ss1 << "/";
    ss1 << argv[1];
//    ss1 << "_nsl";
    ss1 << ".in";

    // freopen(ss1.str().c_str(), "r", stdin);
    FILE* newStdin = freopen(ss1.str().c_str(), "r", stdin);

    // 检查freopen是否成功
    if (newStdin == NULL) {
        std::cerr << "Error: Failed to reopen stdin with file " << ss1.str() << std::endl;
        // 在这里处理错误，比如退出程序或者使用其他方法
        return 1; // 表示程序异常退出
    }

    num_of_walks = atoi(argv[2]);
    seq_len = atoi(argv[3]);

    ss2.str("");
    ss2 << "../path_data/";
    // ss2 << "/data/syf/rw/";
    ss2 << argv[1];
    ss2 << "/";
    ss2 << argv[1];
    ss2 << "_";
    ss2 << argv[2];
    ss2 << "_";
    ss2 << argv[3];
//    ss2 << "_";
//    ss2 << "nsl";
    ss2 << "_merw.txt";

    cout << "File input: " << ss1.str().c_str() << endl;
    cout << "File output: " << ss2.str().c_str() << endl;

    freopen(ss2.str().c_str(), "w", stdout);
    srand(time(0));
    scanf("%d%d", &n, &m);

    cerr << argv[1] << ": " << n << endl;

    cerr << "total tuples: " << m << endl;

    for (int i = 1; i <= m; i++)
    {
        int u, v, t;
        double p;
        scanf("%d%d%d%lf", &u, &v, &t, &p);
        // E存边、p_merw存熵值。邻接表E和边的熵值表p_merw
        link(u, v, t, p);
    }

    /*在时态知识图数据集的训练集中，可能存在不是所有节点都在训练集中出现，
    这些未出现的节点则会在测试集和验证集中出现。
    目前代码中的计算路径和最短距离会出现错误。*/
    for (int i=0;i<n;i++) {
        AT[i].init(E[i], p_merw[i]);
    }

    for (int i = 0; i < n; i++)
        bfs(i);

    /*进行1000轮迭代，每轮迭代对每个节点进行num_of_walks次图路径游走，
    每次游走长度为seq_len，并输出游走路径。*/

    // int totalEpochs = 1000;
    int totalEpochs = 1;
    int progress = 0;
    for (int epoch = 0; epoch < totalEpochs; epoch++)
    {
        for (int st = 0; st < n; st++)
        {
            for (int i = 0; i < num_of_walks; i++)
            {
                int u = st;
                int prior = st;
                int index = 0;
                int cur_time = -1;
                printf("[");
                for (int _ = 0; _ < seq_len; _++)
                {
                    printf("%d", u);
//                    o[_] = dis[st][u];
                    if (_ != 0){
                        index = findIndex(E[prior], u);
                        if (index == -1){
                            index = findIndex(E[u], prior);
                        }
                        if (index == -1){
                             timestamps[0] = -1;
                             t[_] = 1; //
                             timestamps[_] = -1;  // no path -> exapmle: [7120, 7120, 7120, -1, -1 , -1, 1, 1, 1] ,因为-1不是正常timestamp,便于后续判断
                             // continue; 通过这条语句可以判断上面这句话，因为当全是自己时会（没有标点）[25, 25252525250.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
                        }else{
                            if(_ == 1){
                                cur_time = T[prior][index];
                                timestamps[0] = cur_time;
                            }
                            t[_] = exp(-0.1 * abs(cur_time - T[prior][index])) * p_merw[prior][index] + std::sin(T[prior][index]); //时间衰减 * 信息熵 表示节点重要程度
                            timestamps[_] = T[prior][index];
                        }

                    }else{
                        t[_] = 1;
                    }
                    printf(", ");
                    int g = AT[u].roll();
                    if(g == -1){
                        continue; // 例如，如果7120不存在，则会形成路径[7120, 7120, 7120, 7120, 0, 0, 0, 0]输出
                    }
                    prior = u;
                    u = g;
                }

//                for (int _ = 0; _ < seq_len; _++)
//                {
//                    printf("%d", o[_] - 1);
//                    // if (_ != seq_len - 1)
//                    printf(", ");
//                }

                for (int _ = 0; _ < seq_len; _++)
                {
                    printf("%d", timestamps[_]);
                    printf(", ");
                }
                for (int _ = 0; _ < seq_len; _++)
                {
                    // printf("%d", o[_] - 1);
                    printf("%.4f", t[_]);
                    if (_ != seq_len - 1)
                        printf(", ");
                }

                printf("]\n");
            }
        }

        // 计算进度
        progress = static_cast<int>((static_cast<float>(epoch) / totalEpochs) * 100);
        
        // 输出进度条
        cerr << "[";
        for (int i = 0; i < 50; i++) {
            if (i < progress / 2) {
                cerr << "=";
            } else {
                cerr << " ";
            }
        }
        cerr << "] " << progress << "%\r";

    }
    cerr << endl;
    fclose(stdin);
    fclose(stdout);
    return 0;
}