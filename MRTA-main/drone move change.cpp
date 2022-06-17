#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cassert>

#include <algorithm>
#include <random>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <numeric>
#include <string>
#include <queue>
#include <tuple>

constexpr int INFINITE = std::numeric_limits<int>::max();

/**
 *  change simulation settings
 *  TIME_MAX : total time steps
 *	map generated will be MAP_SIZE x MAP_SIZE grid
 *	NUM_ROBOTS : number of robots in simulation
 *  NUM_RTYPE : number of robot types (currently DRONE, CATERPILLAR , WHEEL)
 *  NUM_MAX_TASKS : total number of tasks to be done
 *  NUM_INITIAL_TASKS : number of tasks generated at the beginning of simulation
 *	MAX_ENERGY : total energy that a single robot has at the beginning of simulation
 *  WALL_DENSITY : probability of a grid coordinate being a wall
 *  ENERGY_CONSUMPTION_PER_TICK : energy consumed every timestep when working, or moving
 *
 *  SEED : random seed. with same seed, simulator will generate exactly same random results including (map, object, tasks, actions etc.)
 *  SIMULATOR_VERBOSE : if true, print out maps
 */
constexpr int MAP_SIZE = 40;
constexpr int TIME_MAX = MAP_SIZE * 100;
constexpr int NUM_ROBOT = 6;
constexpr int NUM_RTYPE = 3;
constexpr int NUM_MAX_TASKS = 16;
constexpr int NUM_INITIAL_TASKS = NUM_MAX_TASKS / 2;
constexpr int MAX_ENERGY = TIME_MAX * 6;
constexpr int WALL_DENSITY = 10;
constexpr int ENERGY_CONSUMPTION_PER_TICK = 10;
constexpr int TASK_PROGRESS_PER_TICK = ENERGY_CONSUMPTION_PER_TICK;

constexpr unsigned int SEED = 1;
constexpr bool SIMULATOR_VERBOSE = true;

enum ObjectType : int
{
	UNKNOWN = -1,
	EMPTY = 0b0000,
	ROBOT = 0b0001,
	TASK = 0b0010,
	ROBOT_AND_TASK = 0b0011,
	WALL = 0b0100
};
ObjectType operator&(ObjectType lhs, ObjectType rhs) { return static_cast<ObjectType>(static_cast<int>(lhs) & static_cast<int>(rhs)); }
ObjectType operator|(ObjectType lhs, ObjectType rhs) { return static_cast<ObjectType>(static_cast<int>(lhs) | static_cast<int>(rhs)); }
ObjectType &operator|=(ObjectType &lhs, ObjectType rhs) { return lhs = lhs | rhs; }
ObjectType &operator&=(ObjectType &lhs, ObjectType rhs) { return lhs = lhs & rhs; }
ObjectType &operator&=(ObjectType &lhs, unsigned long rhs) { return lhs = static_cast<ObjectType>(static_cast<int>(lhs) & rhs); }

enum RobotType
{
	INVALID = -1,
	DRONE,
	CATERPILLAR,
	WHEEL,
	OBJECT,
	NUMBER
};
constexpr RobotType robot_types[3] = {RobotType::DRONE, RobotType::CATERPILLAR, RobotType::WHEEL};
std::string robot_names[3] = {"DRONE", "CATERPILLAR", "WHEEL"};

int terreinMatrix[NUM_RTYPE][MAP_SIZE][MAP_SIZE];
int objectMatrix[MAP_SIZE][MAP_SIZE];
int numRobotMatrix[MAP_SIZE][MAP_SIZE];

int knownTerrein[NUM_RTYPE][MAP_SIZE][MAP_SIZE];
int knownObject[MAP_SIZE][MAP_SIZE];

class Coord
{
public:
	int x;
	int y;

	constexpr Coord() : x{-1}, y{-1} {}
	constexpr Coord(int xx, int yy) : x{xx}, y{yy} {}

	friend std::ostream &operator<<(std::ostream &o, const Coord &coord)
	{
		o << "{" << coord.x << ", " << coord.y << "}";
		return o;
	}
};

Coord operator+(const Coord &lhs, const Coord &rhs)
{
	return {lhs.x + rhs.x, lhs.y + rhs.y};
}

Coord operator-(const Coord &lhs, const Coord &rhs)
{
	return {lhs.x - rhs.x, lhs.y - rhs.y};
}

bool operator==(const Coord &lhs, const Coord &rhs)
{
	return lhs.x == rhs.x && rhs.y == lhs.y;
}

bool operator!=(const Coord &lhs, const Coord &rhs)
{
	return !(lhs == rhs);
}

namespace std
{
	template <>
	struct hash<Coord>
	{
		std::size_t operator()(const Coord &c) const noexcept
		{
			unsigned long long hash = c.x << 4;
			hash |= c.y;
			return hash;
		}
	};
} // namespace std

class Robot;
class Task;
class TaskView;

/*
 * @brief Returns the reference of the object at a specified position.
 *
 * @param coord The position.
 * @return ObjectType& The reference of the object.
 */
ObjectType &object_at(Coord coord)
{
	int *ptr = *(objectMatrix + coord.x) + coord.y;
	ObjectType *casted = reinterpret_cast<ObjectType *>(ptr);

	return *casted;
}

ObjectType &known_object_at(Coord coord)
{
	int *ptr = *(knownObject + coord.x) + coord.y;
	ObjectType *casted = reinterpret_cast<ObjectType *>(ptr);

	return *casted;
}

int &terrein_cost_at(RobotType type, Coord coord) { return *(*(*(terreinMatrix + static_cast<int>(type)) + coord.x) + coord.y); }

int &known_terrein_cost_at(RobotType type, Coord coord) { return *(*(*(knownTerrein + static_cast<int>(type)) + coord.x) + coord.y); }

int &num_robots_at(Coord coord) { return *(*(numRobotMatrix + coord.x) + coord.y); }

Coord get_random_empty_position();

void reveal_square_range(Coord centre, int view_range, std::unordered_map<Coord, Task *> &uncharted_tasks, std::vector<TaskView> &active_tasks);

void reveal_cross_range(Coord center, int view_range, std::unordered_map<Coord, Task *> &uncharted_tasks, std::vector<TaskView> &active_tasks);

std::chrono::high_resolution_clock::time_point get_now() { return std::chrono::high_resolution_clock::now(); }

void generateMap(Robot *robots, Task *all_tasks, std::unordered_map<Coord, Task *> &uncharted_tasks, std::vector<TaskView> &active_tasks);

void printMap(int);
void printKnownMap(int type);

void printObjects(const Robot *robots, const Task *all_tasks, int num_tasks);

static const int task_num_length = std::to_string(NUM_MAX_TASKS).length();

class Task
{
	static constexpr int DRONE_DEFAULT_COST = MAX_ENERGY;
	static constexpr int CATERPILLAR_COST_EXCLUSIVE_UPPER_BOUND = 100;
	static constexpr int CATERPILLAR_COST_MIN = 50;
	static constexpr int WHEEL_COST_EXCLUSIVE_UPPER_BOUND = 200;
	static constexpr int WHEEL_COST_MIN = 0;

public:
	Coord coord;
	int id;
	bool done;
	int assigned_robot_id;

	/*
	 * @brief The default constructor that constructs an invalid task.
	 */
	Task()
		: id{-1}, done{false}
	{
		// std::fill(taskCost_, taskCost_ + NUM_RTYPE, INFINITE);
	}

	Task(Coord position, int id, std::initializer_list<int> costs)
		: coord{position}, id{id}, done{false}
	{
		assert(costs.size() == NUM_RTYPE);
		std::copy(costs.begin(), costs.end(), taskCost_);
	}

	int get_cost_by_type(RobotType type) const
	{
		assert(("Requires a valid robot type.", type != RobotType::INVALID));
		return taskCost_[type];
	}

	static Task generate_random(int id)
	{
		Coord position = get_random_empty_position();

		return Task(position, id,
					// Costs
					{
						DRONE_DEFAULT_COST,
						rand() % CATERPILLAR_COST_EXCLUSIVE_UPPER_BOUND + CATERPILLAR_COST_MIN,
						rand() % WHEEL_COST_EXCLUSIVE_UPPER_BOUND + WHEEL_COST_MIN});
	}

	friend std::ostream &operator<<(std::ostream &o, const Task &task)
	{
		o << "Task[" << std::setw(task_num_length) << task.id << "]: ";
		o << "Location " << task.coord << "\t";
		if (task.done)
			o << "Done (by Robot " << task.assigned_robot_id << ")";
		else
			o << "Not done";
		return o;
	}

private:
	int taskCost_[NUM_RTYPE];
};

class TaskView
{
public:
	int id() const noexcept { return task_->id; }
	/*
	 * @brief Returns the true position of this task.
	 */
	Coord coord() const noexcept { return task_->coord; }
	/*
	 * @brief Get the cost by a specified robot type.
	 */
	int get_cost_by_type(RobotType type) const noexcept { return task_->get_cost_by_type(type); }

	TaskView(const Task &task) : task_{std::addressof(task)} {}
	friend void reveal_square_range(Coord centre, int view_range, std::vector<TaskView> &active_tasks);
	friend void reveal_cross_range(Coord center, int view_range, std::vector<TaskView> &active_tasks);

private:
	const Task *task_;
};

struct DispatchResult
{
	bool success;
	Task task;
};

class TaskDispatcher
{
public:
	TaskDispatcher()
		: next_task_id_{NUM_INITIAL_TASKS}, next_task_arrival_time_{TIME_MAX / 4}
	{
	}

	DispatchResult try_dispatch(int current_time)
	{
		if (current_time == next_task_arrival_time_ && next_task_id_ < NUM_MAX_TASKS)
		{
			if (SIMULATOR_VERBOSE)
				std::cout << "At time " << current_time << ": ";

			next_task_arrival_time_ += (TIME_MAX / 2) / (NUM_MAX_TASKS / 2);

			Task new_task{Task::generate_random(next_task_id_)};

			next_task_id_++;

			return {true, new_task};
		}
		else
		{
			return {false, {}};
		}
	}

	int next_task_id() const noexcept { return next_task_id_; }
	int num_remaining_tasks() const noexcept { return NUM_MAX_TASKS - next_task_id_; }

private:
	int next_task_id_;
	int next_task_arrival_time_;
};

enum Status
{
	IDLE,
	WORKING,
	MOVING,
	EXHAUSTED
};
static const std::string status_strs[] = {"IDLE", "WORKING", "MOVING", "EXHAUSTED"};

enum Action
{
	UP,
	DOWN,
	LEFT,
	RIGHT,
	HOLD
};

class Robot
{
public:
	/*
	 * @brief Robot ID
	 */
	int id;
	/*
	 * @brief The current position of this robot.
	 */
	Coord coord;

	/*
	 * @brief
	 * 3 different robot types DRONE, CATERPILLAR, WHEEL
	 */
	RobotType type = RobotType::INVALID;
	/*
	 * @brief
	 * 3 different robot status IDLE, WORKING, MOVING
	 */
	int status = 3;
	/*
	 * @brief
	 * robot remaining energy
	 */
	int energy = 0;
	/*
	 * @brief
	 * coordinate where robot moving towards
	 */
	Coord targetCoord;
	/*
	 * @brief
	 * list of pre_action by Drone
	 */

	void move_step() noexcept
	{
		moving_progress_ -= ENERGY_CONSUMPTION_PER_TICK;
		if (moving_progress_ < 0)
			moving_progress_ = 0;

		consume_energy();
	}

	Task &current_task() const
	{
		assert(("Cannot get the current task: the Robot is not working.", current_task_ != nullptr));
		return *current_task_;
	}

	const std::vector<Task> &finished_tasks() const noexcept { return finished_tasks_; }

	/*
	 * @brief Check if this robot reachs a unfinished task.
	 *
	 * @return int Returns the index of the found task.
	 */
	int is_at_task(const std::vector<TaskView> &active_tasks) const noexcept
	{
		for (auto &task_view : active_tasks)
			if (task_view.coord() == coord)
				return task_view.id();
		return -1;
	}

	/*
	 * @brief Set the task object
	 *
	 * @param task
	 */
	void set_task(Task &task)
	{
		assert(static_cast<int>(type) >= 0);
		status = Status::WORKING;
		task_progress_ = task.get_cost_by_type(type);
		current_task_ = &task;
	}

	/*
	 * @brief is robot is working on a task, reduce its task progress by task_progress_per tick
	 * if robot has cosumed all of its energy, robot status becomes EXHAUSTED
	 *
	 * @return true if robot has remaining energy or the task has been completed.
	 * @return false if robot has no remaining energy but task has not been completed.
	 */
	bool do_task() noexcept
	{
		assert(current_task_ != nullptr);

		task_progress_ -= TASK_PROGRESS_PER_TICK;
		if (task_progress_ < 0)
			task_progress_ = 0;

		consume_energy();

		if (status == EXHAUSTED)
		{
			// Exception: exhausted and finished the current task at the same time.
			if (task_progress_ == 0)
			{
				status = WORKING;
				return true;
			}

			return false;
		}
		return true;
	}

	/*
	 * @brief if task is finished, depending on remaining energy, robot status becomes idle or exhasusted
	 *	id of current task is added to finished task list
	 *
	 */
	void finish_task() noexcept
	{
		status = energy == 0 ? EXHAUSTED : IDLE;
		current_task().done = true;
		finished_tasks_.push_back(*current_task_);
		current_task_ = nullptr;
	}

	/*
	 * @brief
	 *
	 */
	void remove_task() noexcept
	{
		current_task_ = nullptr;
	}

	/*
	 * @brief Set the target coordinate object
	 *
	 * @param action
	 * @param verbose
	 */
	void set_target_coordinate(Action action, bool verbose = false)
	{
		static const Coord actions[5] =
			{
				Coord{0, 1},
				Coord{0, -1},
				Coord{-1, 0},
				Coord{1, 0},
				Coord{0, 0}};

		assert(static_cast<int>(action) < 5 && static_cast<int>(action) >= 0);

		targetCoord = coord + actions[static_cast<int>(action)];

		if (object_at(targetCoord) == WALL ||
			targetCoord.x < 0 ||
			targetCoord.y < 0 ||
			targetCoord.x >= MAP_SIZE ||
			targetCoord.y >= MAP_SIZE ||
			action == HOLD)
		{
			// Invalid target.
			status = IDLE;
			targetCoord = coord;
			return;
		}

		set_travel_cost();

		status = Status::MOVING;
		return;
	}

	/*
	 * @brief returns moving_progress value (see private sections for moving_progress info)
	 *
	 * @return int
	 */
	int remaining_moving_progress() const noexcept { return moving_progress_; }

	/*
	 * @brief returns task_progress value (see private sections for task_progress info)
	 *
	 * @return int
	 */
	int remaining_task_progress() const noexcept { return task_progress_; }

	bool is_exhausted() const noexcept { return status == Status::EXHAUSTED; }

	void set_travel_cost() noexcept { moving_progress_ = get_travel_cost() / 2; }

	void reveal_observed_area(std::unordered_map<Coord, Task *> &uncharted_tasks, std::vector<TaskView> &active_tasks)
	{
		if (is_exhausted())
			return;

		if (type == RobotType::DRONE)
		{
			constexpr int DRONE_VIEW_RANGE = 2;
			reveal_square_range(coord, DRONE_VIEW_RANGE, uncharted_tasks, active_tasks);
		}
		else if (type == RobotType::CATERPILLAR)
		{
			constexpr int CATERPILLAR_VIEW_RANGE = 1;
			reveal_square_range(coord, CATERPILLAR_VIEW_RANGE, uncharted_tasks, active_tasks);
		}
		else if (type == RobotType::WHEEL)
		{
			constexpr int WHEEL_VIEW_RANGE = 1;
			reveal_cross_range(coord, WHEEL_VIEW_RANGE, uncharted_tasks, active_tasks);
		}
	}

	/*
	 * @brief Construct a default Robot object
	 */
	Robot()
		: id{-1}, coord{0, 0}, targetCoord{MAP_SIZE + 1, MAP_SIZE + 1}, energy{MAX_ENERGY}, status{IDLE}, type{RobotType::INVALID} {}

	static Robot create_new(int id, Coord position, RobotType type)
	{
		return Robot(id, position, type);
	}

private:
	void consume_energy() noexcept
	{
		energy -= ENERGY_CONSUMPTION_PER_TICK;
		if (energy <= 0)
		{
			energy = 0;
			status = Status::EXHAUSTED;
		}
	}

	int get_travel_cost() const noexcept { return terreinMatrix[type][coord.x][coord.y]; }
	// int get_task_cost() const noexcept { return assignedList[current_working_task_idx].get_cost_by_type(type); }

	Robot(int id, Coord position, RobotType type)
		: id{id}, coord{position}, targetCoord{position}, energy{MAX_ENERGY}, status{IDLE}, type{type} {}

	/*
	 * progress value is used to track how much the moving/task is complete
	 *
	 * moving_progress is set to half of the cost used to pass the current coordinate and when it reaches zero, coordinate of the moving robot changes.
	 *	then, set to half of the cost used to pass this coordinate and moves until it reaches center of the coordinate grid
	 *
	 *  task_progress is set to the taks cost and used task is complete when it reaches zero.
	 */

	int moving_progress_ = 0;
	int task_progress_ = 0;
	Task *current_task_ = nullptr;
	std::vector<Task> finished_tasks_;
};

struct Scheduler
{
	/*
	 * @brief change of schedule can occur when the map information is updated
	 *
	 * @param known_objects : object(wall, robots, tasks) matrix information available currently
	 * @param known_terrein : cost consumed at coordinate for each robot
	 * @param active_tasks :
	 * @param robot_list : list of robots in simulator
	 */
	virtual void on_info_updated(const int (&known_objects)[MAP_SIZE][MAP_SIZE],
								 const int (&known_terrein)[NUM_RTYPE][MAP_SIZE][MAP_SIZE],
								 const std::vector<TaskView> &active_tasks,
								 const Robot (&robot_list)[NUM_ROBOT]) {}

	/*
	 * @brief decide whether to start working on the task at the coordinate
	 *
	 * @param known_objects : object(wall, robots, tasks) matrix information available currently
	 * @param known_terrein : cost consumed at coordinate for each robot
	 * @param active_tasks :
	 * @param robot_list : list of robots in simulator
	 * @param task : the current task.
	 * @return true : start working on task available at the coordinate
	 * @return false : don't satrt working on the task at the coordinate
	 */
	virtual bool on_task_reached(const int (&known_objects)[MAP_SIZE][MAP_SIZE],
								 const int (&known_terrein)[NUM_RTYPE][MAP_SIZE][MAP_SIZE],
								 const std::vector<TaskView> &active_tasks,
								 const Robot (&robot_list)[NUM_ROBOT],
								 const Task &task,
								 const Robot &current_robot)
	{
		if (current_robot.type == RobotType::DRONE || current_robot.type == RobotType::INVALID)
			return false;
		else
			return true;
	}

	/*
	 * @brief decide which coordinate the robot will move towards (up, down, left, right)
	 *
	 * @param known_objects : object(wall, robots, tasks) matrix information available currently
	 * @param known_terrein : cost consumed at coordinate for each robot
	 * @param active_tasks :
	 * @param robot_list : list of robots in simulator
	 * @param current_robot : robot with this id will have its direction determined by this function
	 * @return int
	 */
	virtual Action calculate_idle_action(const int (&known_objects)[MAP_SIZE][MAP_SIZE],
										 const int (&known_terrein)[NUM_RTYPE][MAP_SIZE][MAP_SIZE],
										 const std::vector<TaskView> &active_tasks,
										 const Robot (&robot_list)[NUM_ROBOT],
										 const Robot &current_robot,
										 int current_time) = 0;
};

/*
 * @brief Scheduling algorithms can be applied by modifying functions below.
 * funtion information available above
 */
class MyScheduler : public Scheduler // TODO: DRONE 외의 로봇들이 초기 TASK 할당 안 됐을때 어떻게 활동할지, time 1/4 지난 시점에서
{
public:
	Action pre_action[2 * NUM_ROBOT] = {HOLD, HOLD, HOLD, HOLD, HOLD, HOLD, HOLD, HOLD, HOLD, HOLD, HOLD, HOLD};
	int robot_allocate_task[NUM_ROBOT] = {-1, -1, -1, -1, -1, -1};

	template <class C, typename T>
	bool contains(C &&c, T t) { return std::find(std::begin(c), std::end(c), t) != std::end(c); }

	bool check_range_over(Coord target)
	{
		return (known_object_at(target) == WALL ||
				target.x < 0 ||
				target.y < 0 ||
				target.x >= MAP_SIZE ||
				target.y >= MAP_SIZE)
				   ? true
				   : false;
	}

	bool check_range_over_drone_x(Coord target)
	{
		return (known_object_at(target) == WALL ||
				target.x < 2 ||
				target.x >= MAP_SIZE - 2)
				   ? true
				   : false;
	}
	bool check_range_over_drone_y(Coord target)
	{
		return (known_object_at(target) == WALL ||
				target.y < 2 ||
				target.y >= MAP_SIZE - 2)
				   ? true
				   : false;
	}

	const Coord actions[5] =
		{
			Coord{0, 1},
			Coord{0, -1},
			Coord{-1, 0},
			Coord{1, 0},
			Coord{0, 0}};

	// Coord prev_visited[NUM_ROBOT];

	Coord drone_target_coord[2];
	bool check_relative_position = false;
	bool check_direction[2] = {false, false};

	void on_info_updated(const int (&known_objects)[MAP_SIZE][MAP_SIZE],
						 const int (&known_terrein)[NUM_RTYPE][MAP_SIZE][MAP_SIZE],
						 const std::vector<TaskView> &active_tasks,
						 const Robot (&robot_list)[NUM_ROBOT]) override
	{
		// for(auto it : active_tasks)
		// {
		// 	std::cout << it.id() << ":"<< it.coord() << ", ";
		// }
		// std::cout<<std::endl;

		//두개의 DRONE 사이의 상대적 위치 결정
		if (!check_relative_position)
		{
			if (robot_list[0].coord.y >= robot_list[3].coord.y && robot_list[0].coord.x >= robot_list[3].coord.x)
			{
				pre_action[robot_list[0].id * 2 + 0] = UP;
				pre_action[robot_list[0].id * 2 + 1] = LEFT;
				pre_action[robot_list[3].id * 2 + 0] = DOWN;
				pre_action[robot_list[3].id * 2 + 1] = RIGHT;
			}
			else if (robot_list[0].coord.y >= robot_list[3].coord.y && robot_list[0].coord.x < robot_list[3].coord.x)
			{
				pre_action[robot_list[0].id * 2 + 0] = UP;
				pre_action[robot_list[0].id * 2 + 1] = RIGHT;
				pre_action[robot_list[3].id * 2 + 0] = DOWN;
				pre_action[robot_list[3].id * 2 + 1] = LEFT;
			}
			else if (robot_list[0].coord.y < robot_list[3].coord.y && robot_list[0].coord.x >= robot_list[3].coord.x)
			{
				pre_action[robot_list[0].id * 2 + 0] = DOWN;
				pre_action[robot_list[0].id * 2 + 1] = LEFT;
				pre_action[robot_list[3].id * 2 + 0] = UP;
				pre_action[robot_list[3].id * 2 + 1] = RIGHT;
			}
			else
			{
				pre_action[robot_list[0].id * 2 + 0] = DOWN;
				pre_action[robot_list[0].id * 2 + 1] = RIGHT;
				pre_action[robot_list[3].id * 2 + 0] = UP;
				pre_action[robot_list[3].id * 2 + 1] = LEFT;
			}
			check_relative_position = true;
		}

		int i;

		for (i = 0; i < 2; i++)
		// for (i = 0; i < 1; i++)
		{
			Coord temp = drone_target_coord[i] - robot_list[i * NUM_RTYPE].coord;
			int len = std::abs(temp.x) + std::abs(temp.y);

			if (robot_list[i * NUM_RTYPE].coord == drone_target_coord[i] ||
				drone_target_coord[i] == Coord{-1, -1} ||
				((known_object_at(drone_target_coord[i]) == WALL) && (len < 3)))
			{
				int j;
				if (!check_direction[i]) // 수평방향으로 이동 설정
				{
					if (pre_action[robot_list[i * NUM_RTYPE].id * 2 + 1] == LEFT)
					{
						for (j = 0; j < robot_list[i * NUM_RTYPE].coord.x; j++)
						{
							if (!check_range_over_drone_x(Coord{j, robot_list[i * NUM_RTYPE].coord.y}))
							{
								drone_target_coord[i].x = j;
								drone_target_coord[i].y = robot_list[i * NUM_RTYPE].coord.y;
								pre_action[robot_list[i * NUM_RTYPE].id * 2 + 1] = RIGHT;
								break;
							}
						}
						if (j == robot_list[i * NUM_RTYPE].coord.x)
						{
							for (j = MAP_SIZE - 1; j > robot_list[i * NUM_RTYPE].coord.x; j--)
							{
								if (!check_range_over_drone_x(Coord{j, robot_list[i * NUM_RTYPE].coord.y}))
								{
									drone_target_coord[i].x = j;
									drone_target_coord[i].y = robot_list[i * NUM_RTYPE].coord.y;
									pre_action[robot_list[i * NUM_RTYPE].id * 2 + 1] = LEFT;
									break;
								}
							}
						}
					}
					else
					{
						for (j = MAP_SIZE - 1; j > robot_list[i * NUM_RTYPE].coord.x; j--)
						{
							if (!check_range_over_drone_x(Coord{j, robot_list[i * NUM_RTYPE].coord.y}))
							{
								drone_target_coord[i].x = j;
								drone_target_coord[i].y = robot_list[i * NUM_RTYPE].coord.y;
								pre_action[robot_list[i * NUM_RTYPE].id * 2 + 1] = LEFT;
								break;
							}
						}
						if (j == robot_list[i * NUM_RTYPE].coord.x)
						{
							for (j = 0; j < robot_list[i * NUM_RTYPE].coord.x; j++)
							{
								if (!check_range_over_drone_x(Coord{j, robot_list[i * NUM_RTYPE].coord.y}))
								{
									drone_target_coord[i].x = j;
									drone_target_coord[i].y = robot_list[i * NUM_RTYPE].coord.y;
									pre_action[robot_list[i * NUM_RTYPE].id * 2 + 1] = RIGHT;
									break;
								}
							}
						}
					}
					check_direction[i] = true;
				}
				else // 수직방향으로 이동 설정
				{
					if (pre_action[robot_list[i * NUM_RTYPE].id * 2 + 0] == UP) //기존에 올라가는 방향으로 탐색중
					{
						for (j = robot_list[i * NUM_RTYPE].coord.y + 5; j < MAP_SIZE; j++)
						{
							if (!check_range_over_drone_y(Coord{robot_list[i * NUM_RTYPE].coord.x, robot_list[i * NUM_RTYPE].coord.y + j}))
							{
								drone_target_coord[i].y = j;
								drone_target_coord[i].x = robot_list[i * NUM_RTYPE].coord.x;
								break;
							}
						}
						if (j == MAP_SIZE)
						{
							for (j = robot_list[i * NUM_RTYPE].coord.y - 5; j > 0; j--)
							{
								if (!check_range_over_drone_y(Coord{robot_list[i * NUM_RTYPE].coord.x, robot_list[i * NUM_RTYPE].coord.y - j}))
								{
									drone_target_coord[i].y = j;
									drone_target_coord[i].x = robot_list[i * NUM_RTYPE].coord.x;
									break;
								}
							}
							pre_action[robot_list[i * NUM_RTYPE].id * 2 + 0] = DOWN;
						}
					}
					else //내려가는 방향으로 진행하고 있을시
					{
						for (j = robot_list[i * NUM_RTYPE].coord.y - 5; j > 0; j--)
						{
							if (!check_range_over_drone_y(Coord{robot_list[i * NUM_RTYPE].coord.x, robot_list[i * NUM_RTYPE].coord.y - j}))
							{
								drone_target_coord[i].y = j;
								drone_target_coord[i].x = robot_list[i * NUM_RTYPE].coord.x;
								break;
							}
						}
						if (j == 0)
						{
							for (j = robot_list[i * NUM_RTYPE].coord.y + 5; j < MAP_SIZE; j++)
							{
								if (!check_range_over_drone_y(Coord{robot_list[i * NUM_RTYPE].coord.x, robot_list[i * NUM_RTYPE].coord.y + j}))
								{
									drone_target_coord[i].y = j;
									drone_target_coord[i].x = robot_list[i * NUM_RTYPE].coord.x;
									break;
								}
							}
							pre_action[robot_list[i * NUM_RTYPE].id * 2 + 0] = UP;
						}
					}
					check_direction[i] = false;
				}
			}
		}

		// std::cout << drone_target_coord[0] << std::endl;

		/***************************Drone외 로봇 할당 작업************************************/
		for (i = 0; i < NUM_ROBOT; i++)
		{
			robot_allocate_task[i] = -1;
		}
		/* 로봇을 기준으로 최저거리 태스크 할당 */
		// for(i=0; i < NUM_ROBOT; i++)
		// {
		// 	int min_id = -1;
		// 	if(robot_list[i].type == DRONE || robot_list[i].energy == 0){continue;}
		// 	int min = INFINITE;
		// 	for(auto it : active_tasks)
		// 	{
		// 		if(contains(robot_allocate_task, it.id())){continue;}
		// 		Coord temp = robot_list[i].coord - it.coord();
		// 		// if(min > temp.x * temp.x + temp.y * temp.y) //직선거리로 비교해서 최소 거리에 있는 Robot에 Task 할당
		// 		// {
		// 		// 	min = temp.x * temp.x + temp.y * temp.y;
		// 		// 	min_id=it.id();
		// 		// }
		// 		if(min > std::abs(temp.x) + std::abs(temp.y)) //이동거리로 비교해서 최소 거리에 있는 Robot에 Task 할당
		// 		{
		// 			min = std::abs(temp.x) + std::abs(temp.y);
		// 			min_id=it.id();
		// 		}
		// 	}
		// 	robot_allocate_task[robot_list[i].id] = min_id;
		// }
		/* TASK을 기준으로 최저거리 로봇에 TASK 할당. */
		for (auto it : active_tasks)
		{
			int min_id = -1;
			int min = INFINITE;
			for (i = 0; i < NUM_ROBOT; i++)
			{
				if (robot_list[i].type == DRONE || robot_list[i].energy == 0)
				{
					continue;
				}
				if (robot_allocate_task[i] != -1)
				{
					continue;
				}
				Coord temp = robot_list[i].coord - it.coord();
				// if(min > temp.x * temp.x + temp.y * temp.y) //직선거리로 비교해서 최소 거리에 있는 Robot에 Task 할당
				// {
				// 	min = temp.x * temp.x + temp.y * temp.y;
				// 	min_id=it.id();
				// }
				if (min > std::abs(temp.x) + std::abs(temp.y)) //이동거리로 비교해서 최소 거리에 있는 Robot에 Task 할당
				{
					min = std::abs(temp.x) + std::abs(temp.y);
					min_id = i;
				}
			}
			robot_allocate_task[min_id] = it.id();
		}
		// std::cout << "robot_allocate_task:" << robot_allocate_task[0] << " " << robot_allocate_task[1] << " " << robot_allocate_task[2] << " " << robot_allocate_task[3] << " " << robot_allocate_task[4] << " " << robot_allocate_task[5] << " " << std::endl;
	}

	bool on_task_reached(const int (&known_objects)[MAP_SIZE][MAP_SIZE],
						 const int (&known_terrein)[NUM_RTYPE][MAP_SIZE][MAP_SIZE],
						 const std::vector<TaskView> &active_tasks,
						 const Robot (&robot_list)[NUM_ROBOT],
						 const Task &task,
						 const Robot &current_robot) override
	{
		return Scheduler::on_task_reached(known_objects, known_terrein, active_tasks, robot_list, task, current_robot);
	}

	Action calculate_idle_action(const int (&known_objects)[MAP_SIZE][MAP_SIZE],
								 const int (&known_terrein)[NUM_RTYPE][MAP_SIZE][MAP_SIZE],
								 const std::vector<TaskView> &active_tasks,
								 const Robot (&robot_list)[NUM_ROBOT],
								 const Robot &current_robot,
								 int current_time) override
	{
		if (current_robot.type == DRONE)
		{
			return A_optimal(known_objects, known_terrein, current_robot, drone_target_coord[current_robot.id / 3]);
			
		}
		else // Robot type != DRONE
		{
			if (robot_allocate_task[current_robot.id] == -1) // allocated task is None
			{
				if (current_time >= (TIME_MAX * 4 / 10) ) 
				{
					std::cout << current_robot.id << " time :" << current_time << std::endl;
					//벽에 부딪힐 시 특정 벡터 방향으로 랜덤으로 튕김
					if (pre_action[current_robot.id * 2 + 0] == HOLD)
					{
						pre_action[current_robot.id * 2 + 0] = static_cast<Action>(rand() % 4);
						pre_action[current_robot.id * 2 + 1] = static_cast<Action>(rand() % 4);
						return pre_action[current_robot.id * 2 + rand() % 2];
					}
					else
					{
						Coord target1 = current_robot.coord + actions[static_cast<int>(pre_action[current_robot.id * 2 + 0])];
						Coord target2 = current_robot.coord + actions[static_cast<int>(pre_action[current_robot.id * 2 + 1])];
						Action target_action = pre_action[current_robot.id * 2 + rand() % 2];
						Coord target = current_robot.coord + actions[static_cast<int>(target_action)];

						if ((known_object_at(target1) == WALL && known_object_at(target2) == WALL) ||
							target.x < 2 ||
							target.y < 2 ||
							target.x >= MAP_SIZE - 2 ||
							target.y >= MAP_SIZE - 2)
						{
							pre_action[current_robot.id * 2 + 0] = HOLD;
							pre_action[current_robot.id * 2 + 1] = HOLD;
							return static_cast<Action>(rand() % 5);
							// return HOLD;
						}
						return target_action;
					}
				}
				return HOLD;
			} 
			else
			{
				Coord target;
				for (auto it : active_tasks)
				{
					if (it.id() == robot_allocate_task[current_robot.id])
					{
						target = it.coord();
						break;
					}
				}
				return A_optimal(known_objects, known_terrein, current_robot, target);
			}
		}

		return HOLD;
		// return static_cast<Action>(rand() % 5);
	}

	Action A_optimal(const int (&known_objects)[MAP_SIZE][MAP_SIZE],
					 const int (&known_terrein)[NUM_RTYPE][MAP_SIZE][MAP_SIZE],
					 const Robot &current_robot, Coord target_coord)
	{
		if (target_coord == current_robot.coord)
		{
			return HOLD;
		}

		Coord temp;
		float F_list[4];
		float min = 100000000;

		int i;

		for (i = 0; i < 4; i++)
		{
			float F = 100000000, G = 0, H = 0;
			temp = current_robot.coord + actions[i];
			if(temp == target_coord)
			{
				return static_cast<Action>(i);
			}
			if (!check_range_over(temp))
			{
				G = static_cast<float>(known_terrein[current_robot.type][temp.x][temp.y]);
				H = find_h(known_terrein, current_robot, temp, target_coord);
				F = G + H;
			}
			F_list[i] = F;
		}
		int action = 4;
		for (i = 0; i < 4; i++)
		{
			if (min > F_list[i])
			{
				min = F_list[i];
				action = i;
			}
		}
		// prev_visited[current_robot.id] = current_robot.coord;
		return static_cast<Action>(action);
	}
	float find_h(const int (&known_terrein)[NUM_RTYPE][MAP_SIZE][MAP_SIZE], const Robot &current_robot, Coord temp, const Coord target_coord)
	{
		int num_matrix = 0;
		float sum = 0;
		int i, j;
		int len, min_x, min_y, max_x, max_y;

		std::tie(len, min_x, min_y, max_x, max_y) = bfs(temp, target_coord);
		if(len == 0 ){return 100000000;}

		num_matrix = (max_x - min_x + 1) * (max_y - min_y + 1);
		for (i = min_x; i <= max_x; i++)
		{
			for (j = min_y; j <= max_y; j++)
			{
				sum += static_cast<float>(known_terrein[current_robot.type][i][j]);
			}
		}
		sum = sum / static_cast<float>(num_matrix) * static_cast<float>(len);
		return sum;
	}
	// float find_h(const int (&known_terrein)[NUM_RTYPE][MAP_SIZE][MAP_SIZE], const Robot &current_robot, Coord temp, const Coord target_coord)
	// {
	// 	int num_matrix = 0;
	// 	float sum = 0;
	// 	int i, j;

	// 	num_matrix = (std::abs(target_coord.x - temp.x) + 1) * (std::abs(target_coord.y - temp.y) + 1);
	// 	for (i = std::min(temp.x, target_coord.x); i <= std::max(temp.x, target_coord.x); i++)
	// 	{
	// 		for (j = std::min(temp.y, target_coord.y); j <= std::max(temp.y, target_coord.y); j++)
	// 		{
	// 			sum += static_cast<float>(known_terrein[current_robot.type][i][j]);
	// 		}
	// 	}
	// 	sum = sum / static_cast<float>(num_matrix) * static_cast<float>(std::abs(target_coord.x - temp.x) + std::abs(target_coord.y - temp.y));
	// 	return sum;
	// }
	std::tuple<int, int, int, int, int> bfs(Coord start, Coord target)
	{
		bool visit[MAP_SIZE][MAP_SIZE]{};
		std::queue<Coord> q;
		std::queue<int> q_len;
		std::queue<int> q_min_xy[2];
		std::queue<int> q_max_xy[2];

		q.push(start);
		q_len.push(0);

		if(start.x <= target.x && start.y <= target.y)
		{
			q_min_xy[0].push(start.x);
			q_min_xy[1].push(start.y);
			q_max_xy[0].push(target.x);
			q_max_xy[1].push(target.y);
		}
		else if(start.x <= target.x && start.y > target.y)
		{
			q_min_xy[0].push(start.x);
			q_min_xy[1].push(target.y);
			q_max_xy[0].push(target.x);
			q_max_xy[1].push(start.y);
		}
		else if(start.x > target.x && start.y <= target.y)
		{
			q_min_xy[0].push(target.x);
			q_min_xy[1].push(start.y);
			q_max_xy[0].push(start.x);
			q_max_xy[1].push(target.y);
		}
		else
		{
			q_min_xy[0].push(target.x);
			q_min_xy[1].push(target.y);
			q_max_xy[0].push(start.x);
			q_max_xy[1].push(start.y);
		}

		visit[start.x][start.y] = true;

		while (!q.empty())
		{
			Coord x = q.front();
			q.pop();

			int len = q_len.front();
			q_len.pop();
			len += 1;

			int min_x = q_min_xy[0].front();
			q_min_xy[0].pop();
			int min_y = q_min_xy[1].front();
			q_min_xy[1].pop();
			int max_x = q_max_xy[0].front();
			q_max_xy[0].pop();
			int max_y = q_max_xy[1].front();
			q_max_xy[1].pop();

			for (int i = 0; i < 4; i++)
			{
				Coord temp = x + actions[i];
				if(min_x > temp.x){min_x = temp.x;}
				if(min_y > temp.y){min_y = temp.y;}
				if(max_x < temp.x){max_x = temp.x;}
				if(max_y < temp.y){max_y = temp.y;}
				if(temp == target)
				{
					return {len, min_x, min_y, max_x, max_y};

				}
				if (!visit[temp.x][temp.y] && !check_range_over(temp))
				{
					visit[temp.x][temp.y] = true;
					q.push(temp);
					q_len.push(len);
					q_min_xy[0].push(min_x);
					q_min_xy[1].push(min_y);
					q_max_xy[0].push(max_x);
					q_max_xy[1].push(max_y);
				}
			}
		}
		return {0,0,0,0,0};
	}
};

int main()
{
	auto time_elapsed = std::chrono::high_resolution_clock::duration::zero();
	auto now = get_now();
	auto update_time_elapsed = [&]()
	{ time_elapsed += (get_now() - now); };

	srand(SEED);
	// Uncomment the line below to get randomised seed.
	srand(time(NULL));

	Robot robots[NUM_ROBOT];
	Task all_tasks[NUM_MAX_TASKS];

	TaskDispatcher task_dispatcher{};

	/*
	 * @brief A list of currently revealed but not started tasks.
	 * uncharted task : 아직 찾지 못 한 task들
	 */
	std::vector<TaskView> active_tasks;
	std::vector<TaskView> schedule_tasks_list;

	std::unordered_map<Coord, Task *> uncharted_tasks;

	/**
	 * The scheduler algorithm for the simulation.
	 * Should change the type of scheulder if you have implemented
	 * algorithms other than `class MyScheduler`.
	 */
	Scheduler &scheduler = *new MyScheduler();

	// Initialise the map
	generateMap(robots, all_tasks, uncharted_tasks, active_tasks);

	// for(auto elem : uncharted_tasks){
	//     std::cout<<"key : "<<elem.first<<" value : "<<elem.second<<std::endl;
	// }

	// prints out maps and object informattion
	if (SIMULATOR_VERBOSE)
	{
		printMap(OBJECT);
		printMap(0);
		printMap(1);
		printMap(2);

		printMap(NUMBER);

		printObjects(robots, all_tasks, task_dispatcher.next_task_id());

		std::cout << "Press Enter to start simulation." << std::endl;
		std::getchar();
	}

	// variable to check if all tasks are done
	bool all_done = false;
	int num_working_robots = 0;
	// variable for time step
	int time = 0;

	while (time < TIME_MAX && not all_done)
	{
		// checks if task is generate on this time step
		DispatchResult result = task_dispatcher.try_dispatch(time);
		// if task is generated
		if (result.success)
		{
			if (SIMULATOR_VERBOSE)
				std::cout << "Task " << result.task.id << " is generated at " << result.task.coord << ".\n";
			// add task to objectmatrix
			object_at(result.task.coord) = TASK;

			auto id = result.task.id;

			// add task to task list and availabe task list
			all_tasks[id] = std::move(result.task);

			uncharted_tasks.insert({result.task.coord, &all_tasks[id]});

			// Update obseravtions from robots.
			for (Robot &robot : robots)
			{
				robot.reveal_observed_area(uncharted_tasks, active_tasks);
			}
		}

		now = get_now();
		// std::cout << "At time " << time << " find target : ";
		scheduler.on_info_updated(knownObject, knownTerrein, active_tasks, robots);
		update_time_elapsed();

		int num_exhausted = 0;
		// simulate robot behavior
		for (int index = 0; index < NUM_ROBOT; index++)
		// for (int index = 0; index < 1; index++)
		{
			Robot &current_robot = robots[index];

			if (current_robot.status == MOVING) // Robot is currently moving
			{
				// Current robot reaches some point.
				if (current_robot.remaining_moving_progress() == 0)
				{
					if (current_robot.coord == current_robot.targetCoord)
					{
						if (SIMULATOR_VERBOSE)
							std::cout << "Robot " << index << " has reached" << current_robot.coord << std::endl;

						// Check whether current robot reaches a task.
						int task_id = current_robot.is_at_task(active_tasks);

						bool start_task = false;

						if (task_id >= 0) // The current robot reachs task of {task_id}.
						{
							Task &task = all_tasks[task_id];

							// object_at(current_robot.coord) = ROBOT_AND_TASK;
							// num_robots_at(current_robot.coord) += 1;

							// decide wether to start working on the task at the coordinate and add algorithm time
							now = get_now();
							start_task = scheduler.on_task_reached(knownObject, knownTerrein, active_tasks, robots, task, current_robot);
							update_time_elapsed();

							if (start_task)
							{
								current_robot.set_task(task);
								task.assigned_robot_id = current_robot.id;

								// Remove the assigned task's id from the active list.
								auto it = std::find_if(active_tasks.begin(), active_tasks.end(),
													   [task_id](const TaskView &view)
													   { return view.id() == task_id; });
								active_tasks.erase(it);
								num_working_robots++;
							}
						}

						// Current robot has decided to not start working on the reached task.
						// Or, there is no task here.
						if (!start_task)
						{
							current_robot.status = IDLE;
						}
					}
					else // Current robot leaves the current position (coordinate).
					{
						if (SIMULATOR_VERBOSE)
						{
							// std::cout << "Robot " << index << " is leaving " << current_robot.coord << std::endl;
						}
						// Make update on object matrix when robotis leaving coordinate
						ObjectType &current_object = object_at(current_robot.coord);

						// Reduce robot count.
						num_robots_at(current_robot.coord) -= 1;
						if (num_robots_at(current_robot.coord) == 0)
						{
							// current_object = current_object == ROBOT
							// 	? EMPTY
							// 	: TASK;
							current_object &= ~(1UL);
						}

						// Move robot to the target coordinate.
						current_robot.coord = current_robot.targetCoord;
						object_at(current_robot.coord) |= ObjectType::ROBOT;
						num_robots_at(current_robot.coord) += 1;

						current_robot.set_travel_cost();

						current_robot.reveal_observed_area(uncharted_tasks, active_tasks);
					}
				}
				// robot has not finished moving yet.
				if (current_robot.remaining_moving_progress() > 0)
				{
					// std::cout << current_robot.id << ", moving: " << current_robot.remaining_moving_progress() << std::endl;
					current_robot.move_step();
				}
			}
			else if (current_robot.status == WORKING) // Robot is currently working on the task
			{
				if (current_robot.remaining_task_progress() > 0) // robot has not finished the task yet
				{
					bool success = current_robot.do_task(); // keep working on the task

					if (!success)
					{
						active_tasks.emplace_back(all_tasks[current_robot.current_task().id]);
						current_robot.remove_task();
					}

					if (SIMULATOR_VERBOSE)
						std::cout << "Robot " << index << " is working on task " << current_robot.current_task().id << ".\n";
					if (current_robot.is_exhausted())
					{
						if (SIMULATOR_VERBOSE)
							std::cout << "Robot " << index << " is exhausted while it is working on task " << current_robot.current_task().id << ".\n";
					}
				}
				else // Task done.
				{
					if (SIMULATOR_VERBOSE)
						std::cout << "Robot " << index << " finished working on task " << current_robot.current_task().id << ".\n";

					object_at(current_robot.coord) = ROBOT;
					current_robot.finish_task();
					--num_working_robots;
				}
			}
			else if (current_robot.status == IDLE) // Robot is currently idle
			{
				// decide which direction to move for the robot and add algorithm time
				now = get_now();
				Action action = scheduler.calculate_idle_action(knownObject, knownTerrein, active_tasks, robots, current_robot, time);
				update_time_elapsed();

				current_robot.set_target_coordinate(action, /*verbose=*/true);
				if (SIMULATOR_VERBOSE && current_robot.targetCoord != current_robot.coord)
				{
					// std::cout << "Robot " << index << " targets " << current_robot.targetCoord << ".\n";
				}
			}
			else if (current_robot.status == EXHAUSTED) // Robot has no remaining energy
			{
				// Do nothing for an exhausted robot. (i.e. energy == 0.)
				++num_exhausted;
			}
			else
			{
				std::cout << "Robot " << index << "status error, end simulation." << std::endl;
				std::terminate();
			}
		}

		// Stop simulation if there is no available robot.
		if (num_exhausted == NUM_ROBOT)
			break;

		++time;

		// End condition.
		all_done = active_tasks.empty() &&
				   uncharted_tasks.empty() &&
				   task_dispatcher.num_remaining_tasks() == 0 &&
				   num_working_robots == 0;
	}

	if (all_done)
		std::cout << "Finished all tasks at time " << time << ".\n";

	//  Report
	for (int index = 0; index < NUM_ROBOT; index++)
		std::cout << "robot " << index << " at (" << robots[index].coord << ") status : " << status_strs[robots[index].status] << " remaining energy: " << robots[index].energy << std::endl;

	std::cout << "Task status: " << std::endl;
	for (auto &task : all_tasks)
	{
		std::cout << "\t- " << task << std::endl;
	}

	std::cout << "Time elapsed running algorithm: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_elapsed).count() << " ms" << std::endl;

	printKnownMap(OBJECT);
	// printKnownMap(0);
	// printKnownMap(1);
	// printKnownMap(2);

	return 0;
}

#pragma region details

void generateMap(Robot *robots, Task *all_tasks, std::unordered_map<Coord, Task *> &uncharted_tasks, std::vector<TaskView> &active_tasks)
{

	// generate walls
	int temp = 0;
	for (int ii = 0; ii < MAP_SIZE; ii++)
	{
		for (int jj = 0; jj < MAP_SIZE; jj++)
		{
			temp = rand() % 100;
			if (temp < WALL_DENSITY)
			{
				objectMatrix[ii][jj] = WALL;
			}
		}
	}

	// generate robots
	for (int ii = 0; ii < NUM_ROBOT; ii++)
	{
		Coord position = get_random_empty_position();
		RobotType type = static_cast<RobotType>(ii % NUM_RTYPE);

		robots[ii] = Robot::create_new(ii, position, type);
		object_at(position) = ROBOT;
		num_robots_at(position) += 1;
	}

	// generate initial tasks
	uncharted_tasks.reserve(NUM_MAX_TASKS);
	for (int ii = 0; ii < NUM_INITIAL_TASKS; ii++)
	{
		all_tasks[ii] = Task::generate_random(ii);
		object_at(all_tasks[ii].coord) = TASK;
		uncharted_tasks.insert({all_tasks[ii].coord, &all_tasks[ii]});
	}

	// generate terrein
	int droneCost = (rand() % 40 + 60) * 2;
	int tempCost = 0;
	for (int ii = 0; ii < MAP_SIZE; ii++)
	{
		for (int jj = 0; jj < MAP_SIZE; jj++)
		{
			if (objectMatrix[ii][jj] == WALL)
			{
				// terreinMatrix[0][ii][jj] = droneCost;
				// terreinMatrix[1][ii][jj] = 300;
				// terreinMatrix[2][ii][jj] = 450;
				// terreinMatrix[0][ii][jj] = 9999;
				// terreinMatrix[1][ii][jj] = 9999;
				// terreinMatrix[2][ii][jj] = 9999;
				terreinMatrix[0][ii][jj] = 210;
				terreinMatrix[1][ii][jj] = 500;
				terreinMatrix[2][ii][jj] = 850;
			}
			else
			{
				terreinMatrix[0][ii][jj] = droneCost;
				tempCost = (rand() % 200);
				terreinMatrix[1][ii][jj] = tempCost * 2 + 100;
				terreinMatrix[2][ii][jj] = tempCost * 4 + 50;
			}
		}
	}

	for (int ii = 0; ii < MAP_SIZE; ii++)
	{
		for (int jj = 0; jj < MAP_SIZE; jj++)
		{
			// for (auto type : robot_types)
			// 	known_terrein_cost_at(type, {ii, jj}) = 2000;
			// known_terrein_cost_at(DRONE, {ii, jj}) = droneCost;
			// known_terrein_cost_at(CATERPILLAR, {ii, jj}) = 300;
			// known_terrein_cost_at(WHEEL, {ii, jj}) = 450;
			known_terrein_cost_at(DRONE, {ii, jj}) = droneCost;
			known_terrein_cost_at(CATERPILLAR, {ii, jj}) = 500;
			known_terrein_cost_at(WHEEL, {ii, jj}) = 850;

			known_object_at({ii, jj}) = UNKNOWN;
		}
	}

	for (int i = 0; i < NUM_ROBOT; ++i)
		robots[i].reveal_observed_area(uncharted_tasks, active_tasks);
}

/*
 *  prints out maps
 *	paramOBJECT
 *	OBJECT : object map
 *	ROBOT TYPE : terrein map for that robot type
 *	NUMBER : map that shows number of robots on that coordinate
 *
 */
void printMap(int type)
{
	if (type == OBJECT)
	{
		printf("Object Map\n\n");
	}
	else if (type == NUMBER)
	{
		printf("Robot count map\n\n");
	}
	else

	{
		printf("Terrein Map for Robot type %d\n\n", type);
	}

	for (int ii = 0; ii < MAP_SIZE; ii++)
	{
		for (int jj = 0; jj < MAP_SIZE; jj++)
		{
			if (type == OBJECT)
			{
				switch (objectMatrix[jj][ii])
				{

				case EMPTY:
					printf("     ");
					break;
				case WALL:
					printf(" WALL");
					break;
				case ROBOT:
					printf(" ROBO");
					break;
				case TASK:
					printf(" TASK");
					break;
				case ROBOT_AND_TASK:
					printf(" RTRT");
					break;
				default:
					printf("error");
					break;
				}
			}
			else if (type == NUMBER)
			{
				if (objectMatrix[jj][ii] == WALL)
				{
					printf(" WALL");
				}
				else
				{
					printf("%4d ", numRobotMatrix[jj][ii]);
				}
			}
			else
			{
				if (objectMatrix[jj][ii] == WALL)
				{
					printf(" WALL");
				}
				else
				{
					printf("%4d ", terreinMatrix[type][jj][ii]);
				}
			}
			if (jj < MAP_SIZE - 1)
			{
				printf("| ");
			}
		}
		printf("\n");
		for (int jj = 0; jj < MAP_SIZE; jj++)
		{
			if (ii < MAP_SIZE - 1)
			{
				printf("-----");
				if (jj < MAP_SIZE - 1)
				{
					printf("o ");
				}
			}
		}
		printf("\n");
	}
	printf("\n");
}

// prints out robots and tasks info
void printObjects(const Robot *robotList, const Task *all_tasks, int num_tasks)
{
	std::cout << "##### Robot Locations #####\n";
	for (int ii = 0; ii < NUM_ROBOT; ii++)
		std::cout << "Robot " << ii
				  << "(" << robot_names[static_cast<int>(robotList[ii].type)]
				  << ") at " << robotList[ii].coord << std::endl;
	std::cout << std::endl;

	std::cout << "##### Tasks #####\n";
	for (int i = 0; i < num_tasks; ++i)
	{
		auto &task = all_tasks[i];

		std::cout << "Task " << std::setw(task_num_length) << task.id
				  << " at " << all_tasks[i].coord << " "
				  << "costs: {"
				  << task.get_cost_by_type(RobotType::DRONE) << ", "
				  << task.get_cost_by_type(RobotType::CATERPILLAR) << ", "
				  << task.get_cost_by_type(RobotType::WHEEL) << "}"
				  << (task.done ? " (Done)" : "")
				  << std::endl;
	}
	std::cout << std::endl;

	// for (auto type : robot_types)
	// {
	// 	std::cout << "Type " << robot_names[static_cast<int>(type)];
	// 	for (auto& task_view : active_tasks)
	// 		std::cout << all_tasks[task_view.id()].get_cost_by_type(type) << " \t";
	// 	std::cout << std::endl;
	// }
	// std::cout << std::endl;
}

Coord get_random_empty_position()
{
	// check if there is any empty place in the map.

	bool valid = false;
	Coord position;
	while (!valid)
	{
		position = {rand() % MAP_SIZE, rand() % MAP_SIZE};
		valid = object_at(position) == ObjectType::EMPTY;
	}

	return position;
}

void reveal_square_range(Coord centre, int view_range, std::unordered_map<Coord, Task *> &uncharted_tasks, std::vector<TaskView> &active_tasks)
{
	const int x_min = std::max(0, centre.x - view_range);
	const int x_max = std::min(MAP_SIZE - 1, centre.x + view_range);
	const int y_min = std::max(0, centre.y - view_range);
	const int y_max = std::min(MAP_SIZE - 1, centre.y + view_range);

	for (int x = x_min; x <= x_max; ++x)
		for (int y = y_min; y <= y_max; ++y)
		{
			const Coord c{x, y};

			known_object_at(c) = object_at(c);
			for (RobotType type : robot_types)
				known_terrein_cost_at(type, c) = terrein_cost_at(type, c);

			if (object_at(c) == ObjectType::TASK)
			{
				auto it = uncharted_tasks.find(c);
				if (it != uncharted_tasks.end())
				{
					active_tasks.emplace_back(*it->second);
					uncharted_tasks.erase(it);
				}
			}
		}
}

void reveal_cross_range(Coord centre, int view_range, std::unordered_map<Coord, Task *> &uncharted_tasks, std::vector<TaskView> &active_tasks)
{
	const int x_min = std::max(0, centre.x - view_range);
	const int x_max = std::min(MAP_SIZE, centre.x + view_range);
	const int y_min = std::max(0, centre.y - view_range);
	const int y_max = std::min(MAP_SIZE, centre.y + view_range);

	for (int x = x_min; x <= x_max; ++x)
	{
		const Coord c{x, centre.y};

		known_object_at(c) = object_at(c);
		for (RobotType type : robot_types)
			known_terrein_cost_at(type, c) = terrein_cost_at(type, c);
		if (object_at(c) == ObjectType::TASK)
		{
			auto it = uncharted_tasks.find(c);
			if (it != uncharted_tasks.end())
			{
				active_tasks.emplace_back(*it->second);
				uncharted_tasks.erase(it);
			}
		}
	}

	for (int y = y_min; y <= y_max; ++y)
	{
		const Coord c{centre.x, y};

		known_object_at(c) = object_at(c);
		for (RobotType type : robot_types)
			known_terrein_cost_at(type, c) = terrein_cost_at(type, c);

		if (object_at(c) == ObjectType::TASK)
		{
			auto it = uncharted_tasks.find(c);
			if (it != uncharted_tasks.end())
			{
				active_tasks.emplace_back(*it->second);
				uncharted_tasks.erase(it);
			}
		}
	}
}

void printKnownMap(int type)
{
	if (type == OBJECT)
	{
		printf("Object Map\n\n");
	}
	else if (type == NUMBER)
	{
		printf("Robot count map\n\n");
	}
	else
	{
		printf("Terrein Map for Robot type %d\n\n", type);
	}

	for (int ii = 0; ii < MAP_SIZE; ii++)
	{
		for (int jj = 0; jj < MAP_SIZE; jj++)
		{
			if (type == OBJECT)
			{
				switch (knownObject[jj][ii])
				{

				case EMPTY:
					printf("     ");
					break;
				case WALL:
					printf(" WALL");
					break;
				case ROBOT:
					printf(" ROBO");
					break;
				case TASK:
					printf(" TASK");
					break;
				case ROBOT_AND_TASK:
					printf(" RTRT");
					break;
				default:
					printf("error");
					break;
				}
			}
			else if (type == NUMBER)
			{
				if (knownObject[jj][ii] == WALL)
				{
					printf(" WALL");
				}
				else
				{
					printf("%4d ", numRobotMatrix[jj][ii]);
				}
			}
			else
			{
				if (knownObject[jj][ii] == WALL)
				{
					printf(" WALL");
				}
				else
				{
					printf("%4d ", knownTerrein[type][jj][ii]);
				}
			}
			if (jj < MAP_SIZE - 1)
			{
				printf("| ");
			}
		}
		printf("\n");
		for (int jj = 0; jj < MAP_SIZE; jj++)
		{
			if (ii < MAP_SIZE - 1)
			{
				printf("-----");
				if (jj < MAP_SIZE - 1)
				{
					printf("o ");
				}
			}
		}
		printf("\n");
	}
	printf("\n");
}

#pragma endregion details
