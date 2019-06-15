import cv2
import numpy as np


def load_collage(path, part_size):
    assert len(part_size) == 2, "part_size should be something like (64, 64)"
    im = cv2.imread(path)
    assert all(imd % partd == 0 for imd, partd in zip(im.shape, part_size))
    rows, cols = tuple(imd // partd for imd, partd in zip(im.shape, part_size))
    height, width = part_size
    for row in range(rows):
        for col in range(cols):
            yield ImagePart(
                im[
                    height * row: height * (row + 1),
                    width * col: width * (col + 1),
                ],
                rows,
                cols,
            )


class ImagePart:
    def __init__(self, im, rows, cols):
        self.rows = rows
        self.cols = cols
        self.im = im
        self.west = None
        self.east = None
        self.north = None
        self.south = None

    def dislodge(self):
        self.west = None
        self.east = None
        self.north = None
        self.south = None

    def upper_left(self):
        if self.west:
            return self.west.upper_left()
        if self.north:
            return self.north.upper_left()
        return self

    def free_edges(self):
        edges = set()
        if self.west is None:
            edges.add('west')
        if self.east is None:
            edges.add('east')
        if self.north is None:
            edges.add('north')
        if self.south is None:
            edges.add('south')
        return edges

    def join_groups(self, coords):
        for coord, part in coords.items():
            row, col = coord
            other = (row, col - 1)
            if other in coords:
                part.west = coords[other]
                coords[other].east = part
            other = (row, col + 1)
            if other in coords:
                part.east = coords[other]
                coords[other].west = part
            other = (row + 1, col)
            if other in coords:
                part.south = coords[other]
                coords[other].north = part
            other = (row - 1, col)
            if other in coords:
                part.north = coords[other]
                coords[other].south = part

    def add(self, west=None, north=None):
        if west and self.west is not None:
            return False
        if north and self.north is not None:
            return False
        rows, cols, coords = self.collage_size(
            west=west, north=north,
        )
        if rows < 0 or rows > self.rows or cols > self.cols:
            return False
        
        self.join_groups(coords)
        return True

    def collage_matrix(self):
        nrows, ncols, visited = self.collage_size()
        matrix = np.zeros((nrows, ncols), dtype=int)
        rows, cols = zip(*visited.keys())
        rmin = min(rows)
        cmin = min(cols)
        matrix = {}
        for row, col in visited:
            matrix[row - rmin, col - cmin] = visited[(row, col)]
        return matrix
        

    def collage_size(self, west=None, north=None):
        other_visited = {}
        visited = {}
        if west:
            west._coords(other_visited, coord=(0, -1))
        elif north:
            north._coords(other_visited, coord=(-1, 0))
        if self in other_visited.values():
            return -1, -1, set()
        self._coords(visited, coord=(0, 0))
        if north in visited.values() or west in visited.values():
            return -1, -1, set()
        if set(other_visited.keys()).intersection(visited.keys()):
            return -1, -1, set()
        visited.update(other_visited)
        rows, cols = zip(*visited)
        nrows = max(rows) - min(rows) + 1
        ncols = max(cols) - min(cols) + 1
        return nrows, ncols, visited 

    def _coords(self, visited, coord=(0, 0)):
        if coord in visited:
            return
        elif self in visited.values():
            print("Warning self is visited with two coords {} and {}".format(
                coord, [c for c, v in visited.items() if v == self],
            ))
            return
        visited[coord] = self
        row, col = coord
        if self.west:
            self.west._coords(visited, coord=(row, col - 1))
        if self.east:
            self.east._coords(visited, coord=(row, col + 1))
        if self.south:
            self.south._coords(visited, coord=(row + 1, col))
        if self.north:
            self.north._coords(visited, coord=(row - 1, col))
        return 

    def get_group(self, group=None):
        if group is None:
            group = set()
        group.add(self)
        if self.west and self.west not in group:
            self.west.get_group(group)
        if self.east and self.east not in group:
            self.east.get_group(group)
        if self.south and self.south not in group:
            self.south.get_group(group)
        if self.north and self.north not in group:
            self.north.get_group(group)
        return group


def rnd_energy(horizontal=None, vertical=None):
    return np.random.random()
    

def edge_distance(horizontal=None, vertical=None):
    if horizontal:
        left, right = horizontal
        left.im[:, -1].size
        return ((left.im[:, -1] - right.im[:, 0]) ** 2).sum() / 2
    else:
        top, bottom = vertical
        n = top.im[-1].size
        return ((top.im[-1] - bottom.im[0]) ** 2).sum() / n


def pair_best(image_parts, *, metric=rnd_energy, join=None):
    free = {
        'east': set(),
        'west': set(),
        'north': set(),
        'south': set(),
    }
    print("Finding free edges")
    for image_part in image_parts:
        for edge in image_part.free_edges():
            free[edge].add(image_part)
    suggestions = []
    for west in free['west']:
        for east in free['east']:
            if west != east:
                if join:
                    if (west in join) == (east in join):
                        continue
                # The one with free east should be to the west
                energy = metric(horizontal=(east, west))
                suggestions.append(((west, east, 'H'), energy))
    print("Evaluating energies")
    for south in free['south']:
        for north in free['north']:
            if south != north:
                if join:
                    if (south in join) == (north in join):
                        continue
                # The one with free south shoud be to the north
                energy = metric(vertical=(south, north))
                suggestions.append(((north, south, 'V'), energy))
    pairs, energies = zip(*suggestions)
    order = np.argsort(energies)
    best_group = image_part
    before_size = len(best_group.get_group()) 
    print("Merging")
    for idx in order:
        a, b, direction = pairs[idx]
        agroup = a.get_group()
        bgroup = b.get_group()
        if join and a not in join and b not in join:
            continue
        if b in agroup:
            continue
        elif agroup.intersection(bgroup):
            print("oops")
            continue
        if direction == 'H':
            success = a.add(west=b)
        else:
            success = a.add(north=b)
        if success:
            join = a.get_group()
            best_group = a

    group = best_group.get_group()
    if len(group) != len(image_parts): 
        for image_part in image_parts:
            if image_part in group:
                continue
            image_part.dislodge()
        if before_size != len(group):
            pair_best(image_parts, metric=metric, join=group)

    return best_group.upper_left()


def to_image_data(part, rows, cols):
    shape = list(part.im.shape)
    shape[0] *= rows
    shape[1] *= cols
    row = 0
    col = 0
    height, width = part.im.shape[:2]
    im = np.zeros(shape, dtype=part.im.dtype)
    matrix = part.collage_matrix()
    for (row, col), part in matrix.items():
        im[
            height * row: height * (row + 1),
            width * col: width * (col + 1),
        ] = part.im
    return im 
        

