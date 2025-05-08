import heapq
from collections import deque


class Edge:
    def __init__(self, to, rev, capacity, cost):
        self.to = to
        self.rev = rev
        self.capacity = capacity
        self.cost = cost


class MinCostFlow:
    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]

    def add_edge(self, fr, to,  cost,cap):
        forward = Edge(to, len(self.graph[to]), cap, cost)
        backward = Edge(fr, len(self.graph[fr]), 0, -cost)
        self.graph[fr].append(forward)
        self.graph[to].append(backward)

    def flow(self, s, t, max_flow):
        flow = 0
        cost = 0
        h = [0] * self.n
        prev_v = [0] * self.n
        prev_e = [0] * self.n

        while flow < max_flow:
            dist = [float('inf')] * self.n
            dist[s] = 0
            in_queue = [False] * self.n
            in_queue[s] = True
            q = deque([s])

            # Improved Bellman-Ford with early termination
            while q:
                v = q.popleft()
                in_queue[v] = False
                for i, edge in enumerate(self.graph[v]):
                    if edge.capacity > 0 and dist[edge.to] > dist[v] + edge.cost + h[v] - h[edge.to]:
                        dist[edge.to] = dist[v] + edge.cost + h[v] - h[edge.to]
                        prev_v[edge.to] = v
                        prev_e[edge.to] = i
                        if not in_queue[edge.to]:
                            q.append(edge.to)
                            in_queue[edge.to] = True

            if dist[t] == float('inf'):
                break

            # Update potentials
            for i in range(self.n):
                if dist[i] < float('inf'):
                    h[i] += dist[i]

            # Calculate actual path cost
            path_cost = 0
            v = t
            path = []
            while v != s:
                edge = self.graph[prev_v[v]][prev_e[v]]
                path_cost += edge.cost
                path.append(v)
                v = prev_v[v]
            path.append(s)
            path.reverse()

            delta_flow = max_flow - flow
            v = t
            while v != s:
                edge = self.graph[prev_v[v]][prev_e[v]]
                delta_flow = min(delta_flow, edge.capacity)
                v = prev_v[v]

            flow += delta_flow
            cost += delta_flow * path_cost  # Use actual path cost

            v = t
            while v != s:
                edge = self.graph[prev_v[v]][prev_e[v]]
                edge.capacity -= delta_flow
                self.graph[v][edge.rev].capacity += delta_flow
                v = prev_v[v]

        return flow, cost
