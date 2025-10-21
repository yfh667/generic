from basicSa.route  import dijstra_route

if __name__ == '__main__':
    visibility_matrix = [
        [0, 0, 0, 0, 1],
        [0, 0, 3, 0, 0],
        [0, 0, 0, 4, 0],
        [0, 0, 0, 0, 5],
        [0, 0, 0, 0, 0]
    ]
    distances, paths = dijstra_route.StationA2B(2, 0, 1, visibility_matrix)
    print(distances)
    print(paths)
