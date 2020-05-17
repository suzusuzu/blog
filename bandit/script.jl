using Random
using Statistics
using StatsBase
using Distributions
using Plots
gr()

mutable struct State
    cnt::Int
    n_arms::Int
    sum_reward::Float64
    cum_reward_history::AbstractArray{Float64}
    cnt_per_arms::AbstractArray{Int}
    mus::AbstractArray{Float64}
end

abstract type Policy end

# RandomSelect
struct RandomSelect <: Policy end

function select(rng::RandomSelect, state::State)::Int
    rand(1:state.n_arms)
end

function update(rng::RandomSelect, state::State, armi, reward)
end

function Base.show(io::IO, rng::RandomSelect)
    print(io, "RandomSelect")
end


# UCB1
struct UCB1 <: Policy
    values::AbstractArray{Float64}
end

function select(ucb::UCB1, state::State)::Int
    if minimum(state.cnt_per_arms) == 0
        return argmin(state.cnt_per_arms)
    end
    bounds = sqrt.(2 * log(state.cnt) ./ state.cnt_per_arms)
    argmax(ucb.values + bounds)
end

function update(ucb::UCB1, state::State, armi, reward)
    n = state.cnt_per_arms[armi]
    ucb.values[armi] = (reward + (n-1) * ucb.values[armi]) / n
end

function Base.show(io::IO, ucb::UCB1)
    print(io, "UCB1")
end


# Softmax
mutable struct Softmax <: Policy
    t::Float64
    values::AbstractArray{Float64}
end

function select(sm::Softmax, state::State)::Int
    if minimum(state.cnt_per_arms) == 0
        return argmin(state.cnt_per_arms)
    end

    xs = [ exp(v/sm.t) for v in sm.values]
    sum_xs = sum(xs)
    ps = [ exp(v/sm.t)/sum_xs for v in sm.values]
    sample(Weights(ps))
end

function update(sm::Softmax, state::State, armi, reward)
    n = state.cnt_per_arms[armi]
    sm.values[armi] = (reward + (n-1) * sm.values[armi]) / n
    # sm.t = 1/log(state.cnt)
    # sm.t = 1000/state.cnt
end

function Base.show(io::IO, sm::Softmax)
    print(io, "Softmax(", sm.t, ")")
end


# EpsilonGreedy
mutable struct EpsilonGreedy <: Policy
    epsilon::Float64
    values::AbstractArray{Float64}
end

function select(eg::EpsilonGreedy, state::State)::Int
    if rand() < eg.epsilon
        rand(1:state.n_arms)
    else
        argmax(eg.values)
    end
end

function update(eg::EpsilonGreedy, state::State, armi, reward)
    n = state.cnt_per_arms[armi]
    eg.values[armi] = (reward + (n-1) * eg.values[armi]) / n
end

function Base.show(io::IO, gb::EpsilonGreedy)
    print(io, "EpsilonGreedy(", gb.epsilon, ")")
end



# ThompsonSampling
mutable struct ThompsonSampling <: Policy
    alpha0::Float64
    beta0::Float64
    mu0::Float64
    v0::Float64
    xs::AbstractVector{AbstractVector{Float64}}
end

function select(ts::ThompsonSampling, state::State)::Int
    samples = []
    for armi in 1:state.n_arms
        d = ts.xs[armi]
        n = length(d)
        mean_d = n == 0 ? 0.0 : mean(d)

        mu = (ts.v0*ts.mu0 + n * mean_d)/(ts.v0 + n)
        v = ts.v0 + n
        alpha = ts.alpha0 + n/2
        beta = ts.beta0 + 1/2*sum([(x - mean_d)^2 for x in d]) + (ts.v0*n*(mean_d - ts.mu0)^2)/(2*(ts.v0 + n))

        gamma = Gamma(alpha, 1/beta)
        lambda_ = rand(gamma)
        normal_mu_ = Normal(mu, sqrt(1/(v * lambda_)) )
        mu_ = rand(normal_mu_)

        push!(samples, mu_)
    end
    argmax(samples)
end

function update(ts::ThompsonSampling, state::State, armi, reward)
    push!(ts.xs[armi], reward)
end

function Base.show(io::IO, ts::ThompsonSampling)
    print(io, "ThompsonSampling")
end


function init_state(n_arms::Int; seed=0)::State
    Random.seed!(seed)
    State(0, n_arms, 0.0, zeros(0), zeros(n_arms), rand(n_arms))
end

function get_reward(state::State, armi)::Float64
    randn() + state.mus[armi]
end

function run(policy::Policy, state::State; n=10000, seed=0)
    Random.seed!(seed)
    for i in 1:n
        state.cnt += 1
        armi = select(policy, state)
        state.cnt_per_arms[armi] += 1
        reward = get_reward(state, armi)
        state.sum_reward += reward
        push!(state.cum_reward_history, state.sum_reward)
        update(policy, state, armi, reward)
    end
end


n_arms = 10
state = init_state(n_arms)

# policy
r = RandomSelect()
eg1 = EpsilonGreedy(0.05, zeros(n_arms))
eg2 = EpsilonGreedy(0.1, zeros(n_arms))
sm1 = Softmax(0.05, zeros(n_arms))
sm2 = Softmax(0.1, zeros(n_arms))
ucb = UCB1(zeros(n_arms))
ts = ThompsonSampling(1,1,0.0,0.00000001,[Vector() for _ in 1:n_arms])
ps = [r eg1 eg2 sm1 sm2 ucb ts]

cum_reward_historys = []
println("Max mu : ", argmax(state.mus))
println("mus : ", state.mus)
for p in ps
    s = deepcopy(state)
    run(p, s)
    println("======================================")
    println("Policy : ", p)
    println("Cumulative Reward : ", s.sum_reward)
    println(s.cnt_per_arms)
    println("Max Select : ", argmax(s.cnt_per_arms))
    push!(cum_reward_historys, s.cum_reward_history)
end

n = length(cum_reward_historys[1])
v = cum_reward_historys[1] ./ range(1, n, step=1)
for i in 2:length(ps)
    global v
    tmp = cum_reward_historys[i] ./ range(1, n, step=1)
    v = hcat(v, tmp)
end

label = map((x) -> string(x), ps)
plot(v, label = label, ylim=(0.3, 1.05), xlabel="Time", ylabel="Cumulative Reward / Time", legend = :bottomright)
savefig("plots.svg")
savefig("plots.png")