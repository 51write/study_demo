from collections import deque
from typing import Dict, Any, List, Optional

from langchain import LLMChain
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.chains.base import Chain
from langchain.llms import BaseLLM
from langchain.vectorstores import VectorStore
from pydantic import BaseModel, Field
from task_agent import TaskCreationChain, TaskPrioritizationChain

def get_next_task(
    task_creation_chain: LLMChain,
    result: Dict,
    task_description: str,
    task_list: List[str],
    objective: str,
) -> List[Dict]:
    """Get the next task."""
    incomplete_tasks = ", ".join(task_list)
    response = task_creation_chain.run(
        result=result,
        task_description=task_description,
        incomplete_tasks=incomplete_tasks,
        objective=objective,
    )
    new_tasks = response.split("\n")
    return [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]


def prioritize_tasks(
    task_prioritization_chain: LLMChain,
    this_task_id: int,
    task_list: List[Dict],
    objective: str,
) -> List[Dict]:
    """Prioritize tasks."""
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id) + 1
    response = task_prioritization_chain.run(
        task_names=task_names, next_task_id=next_task_id, objective=objective
    )
    new_tasks = response.split("\n")
    prioritized_task_list = []
    for task_string in new_tasks:
        if not task_string.strip():
            continue
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            prioritized_task_list.append({"task_id": task_id, "task_name": task_name})
    return prioritized_task_list


def _get_top_tasks(vectorstore, query: str, k: int) -> List[str]:
    """Get the top k tasks based on the query."""
    print(f'k={k},query={query}')
    results = vectorstore.similarity_search_with_score(query, k=k)
    if not results:
        return []
    sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))
    return [str(item.metadata["task"]) for item in sorted_results]


def execute_task(
    vectorstore, execution_chain: LLMChain, objective: str, task: str, k: int = 5
) -> str:
    """Execute a task."""
    context = _get_top_tasks(vectorstore, query=objective, k=k)
    return execution_chain.run(objective=objective, context=context, task=task)


from utils import get_prompt,get_tools
class BabyAGI(Chain, BaseModel):
    """Controller model for the BabyAGI agent."""

    task_list: deque = Field(default_factory=deque)
    task_creation_chain: TaskCreationChain = Field(...)
    task_prioritization_chain: TaskPrioritizationChain = Field(...)
    execution_chain: AgentExecutor = Field(...)
    task_id_counter: int = Field(1)
    vectorstore: VectorStore = Field(init=False)
    max_iterations: Optional[int] = None
    
    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def add_task(self, task: Dict):
        self.task_list.append(task)

    def print_task_list(self):
        yield "\n*****TASK LIST*****\n"
        for t in self.task_list:
            yield str(t["task_id"]) + ": " + t["task_name"]

    def print_next_task(self, task: Dict):
        yield "\n*****NEXT TASK*****\n"
        yield str(task["task_id"]) + ": " + task["task_name"]

    def print_task_result(self, result: str):
        yield "\n*****TASK RESULT*****\n"
        yield result

    @property
    def input_keys(self) -> List[str]:
        return ["objective"]

    @property
    def output_keys(self) -> List[str]:
        return []

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent."""
        objective = inputs["objective"]
        first_task = inputs.get("first_task", "Make a todo list")
        self.add_task({"task_id": 1, "task_name": first_task})
        num_iters = 0
        while True:
            if self.task_list:
                yield from self.print_task_list()

                # Step 1: Pull the first task
                task = self.task_list.popleft()
                yield from self.print_next_task(task)

                # Step 2: Execute the task
                result = execute_task(
                    self.vectorstore, self.execution_chain, objective, task["task_name"]
                )
                this_task_id = int(task["task_id"])
                yield from self.print_task_result(result)

                # Step 3: Store the result in the VectorStore
                result_id = f"result_{task['task_id']}"
                self.vectorstore.add_texts(
                    texts=[result],
                    metadatas=[{"task": task["task_name"]}],
                    ids=[result_id],
                )

                # Step 4: Create new tasks and reprioritize task list
                new_tasks = get_next_task(
                    self.task_creation_chain,
                    result,
                    task["task_name"],
                    [t["task_name"] for t in self.task_list],
                    objective,
                )
                for new_task in new_tasks:
                    self.task_id_counter += 1
                    new_task.update({"task_id": self.task_id_counter})
                    self.add_task(new_task)
                self.task_list = deque(
                    prioritize_tasks(
                        self.task_prioritization_chain,
                        this_task_id,
                        list(self.task_list),
                        objective,
                    )
                )
            num_iters += 1
            if self.max_iterations is not None and num_iters == self.max_iterations:
                yield "\n*****TASK ENDING*****\n"
                break

    @classmethod
    def from_llm(
        cls,
        llm: BaseLLM,
        vectorstore: VectorStore,
        verbose: bool = False,
        **kwargs,
    ) -> "BabyAGI":
        """Initialize the BabyAGI Controller."""
        task_creation_chain = TaskCreationChain.from_llm(llm, verbose=verbose)
        task_prioritization_chain = TaskPrioritizationChain.from_llm(
            llm, verbose=verbose
        )
        tools = get_tools(llm)
        prompt=get_prompt(tools)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True
        )
        return cls(
            task_creation_chain=task_creation_chain,
            task_prioritization_chain=task_prioritization_chain,
            execution_chain=agent_executor,
            vectorstore=vectorstore,
            **kwargs,
        )

from config import set_environment
import os
from langchain_community.chat_models import QianfanChatEndpoint
set_environment()
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_community.embeddings import QianfanEmbeddingsEndpoint

OBJECTIVE = "Find the cheapest price and site to buy a Yubikey 5c online and give me the URL"
qianfan_api_key = os.environ['QIANFAN_API_KEY']
qianfan_secret_key= os.environ['QIANFAN_SECRET_KEY']
llm = QianfanChatEndpoint(api_key=qianfan_api_key, secret_key=qianfan_secret_key,model = 'ERNIE-4.0-Turbo-8K-Latest')

# Initialize the vectorstore as empty
import faiss
embedding_size = 1024##这个值要注意，随着使用的模型变化，会是需要手动调整的
index = faiss.IndexFlatL2(embedding_size)
embeddings = QianfanEmbeddingsEndpoint(qianfan_ak=qianfan_api_key, qianfan_sk=qianfan_secret_key,model='bge-large-en')
vectorstore = FAISS(embeddings.embed_query, index, InMemoryDocstore({}), {})
vectorstore.similarity_search_with_score("搜索", k=5)
# Logging of LLMChains
verbose=False
# If None, will keep on going forever
max_iterations: Optional[int] = 2
baby_agi = BabyAGI.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    verbose=verbose,
    max_iterations=max_iterations
)
baby_agi({"objective": OBJECTIVE})
OBJECTIVE = "Find the of a Yubikey 5c on Amazon and give me the URL"

baby_agi({"objective": OBJECTIVE})

