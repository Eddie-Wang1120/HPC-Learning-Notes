#include <iostream>
#include <sstream>
#include <unordered_map>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace std;

//Lexer


std::unordered_map <std::string, std::string> Token{
    {"tok_eof", "-1"},
    {"tok_def", "-2"},
    {"tok_extern", "-3"},
    {"tok_identifier", "-4"},
    {"tok_number", "-5"},
    {"tok_next", "-6"},
};

std::unordered_map <std::string, int> BinOP{
    {"大于", 10},
    {"加", 20},
    {"减", 20},
    {"乘", 40},
};


static std::string identifier;
static double NumVal;

static bool isNum(std::string str){
    std::stringstream check(str);
    double d;
    char c;
    if(!(check >> d)) return false;
    if(check >> c) return false;
    return true;
}

static std::string gettok(){
    static std::string LastStr = " ";
    while (LastStr == " ")
    {
        std::cin>>LastStr; //divide into one word

    }

    if(isNum(LastStr)){
        std::string NumStr;
        do{
            NumStr += LastStr;
            std::cin>>LastStr;
        }while(isNum(LastStr));

        NumVal = strtod(NumStr.c_str(), nullptr);
        return "tok_number";
    }

    if(LastStr == "函数")
        return "tok_def";

    if(LastStr == "外部")
        return "tok_extern";
    
    if(LastStr == "左界")
        return "(";
    
    if(LastStr == "右界")
        return ")";

    if(LastStr == "注释"){
        do
        {
            std::cin>>LastStr;
        } while (LastStr != "换行");
    }

    if(LastStr == "换行")
        return "tok_next";

    identifier = LastStr;
    return "tok_identifier";

}


namespace{

class ExprAST {
    public:
        virtual ~ExprAST() = default;
};

class NumberExprAST : public ExprAST {
  double Val;

public:
  NumberExprAST(double Val) : Val(Val) {}
};
    
class VariableExprAST : public ExprAST {
  std::string Name;

public:
  VariableExprAST(const std::string &Name) : Name(Name) {}
};

class BinaryExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  BinaryExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}
};

class CallExprAST : public ExprAST {
  std::string Callee;
  std::vector<std::unique_ptr<ExprAST>> Args;

public:
  CallExprAST(const std::string &Callee,
              std::vector<std::unique_ptr<ExprAST>> Args)
      : Callee(Callee), Args(std::move(Args)) {}
};

class PrototypeAST {
  std::string Name;
  std::vector<std::string> Args;

public:
  PrototypeAST(const std::string &Name, std::vector<std::string> Args)
      : Name(Name), Args(std::move(Args)) {}

  const std::string &getName() const { return Name; }
};

class FunctionAST {
  std::unique_ptr<PrototypeAST> Proto;
  std::unique_ptr<ExprAST> Body;

public:
  FunctionAST(std::unique_ptr<PrototypeAST> Proto,
              std::unique_ptr<ExprAST> Body)
      : Proto(std::move(Proto)), Body(std::move(Body)) {}
};
}

static std::string CurTok;
static std::string getNextToken() { return CurTok = gettok();}

static std::map<std::string, int> BinopPrecedence;

static int GetTokPrecedence() {
  // Make sure it's a declared binop.
  int TokPrec = BinopPrecedence[CurTok];
  if (TokPrec <= 0)
    return -1;
  return TokPrec;
}

std::unique_ptr<ExprAST> LogError(const char *Str) {
  fprintf(stderr, "Error: %s\n", Str);
  return nullptr;
}
std::unique_ptr<PrototypeAST> LogErrorP(const char *Str) {
  LogError(Str);
  return nullptr;
}

static std::unique_ptr<ExprAST> ParseExpression();

static std::unique_ptr<ExprAST> ParseNumberExpr() {
  auto Result = std::make_unique<NumberExprAST>(NumVal);
  getNextToken(); // consume the number
  return std::move(Result);
}

static std::unique_ptr<ExprAST> ParseParenExpr() {
  getNextToken(); // eat (.
  auto V = ParseExpression();
  if (!V)
    return nullptr;

  if (CurTok != ")")
    return LogError("expected 右界");
  getNextToken(); // eat ).
  return V;
}


static std::unique_ptr<ExprAST> ParseIdentifierExpr() {
  std::string IdName = identifier;

  getNextToken(); // eat identifier.

  if (CurTok != "(") // Simple variable ref.
    return std::make_unique<VariableExprAST>(IdName);

  // Call.
  getNextToken(); 
  std::vector<std::unique_ptr<ExprAST>> Args;
  if (CurTok != ")") {
    while (true) {
      if (auto Arg = ParseExpression())
        Args.push_back(std::move(Arg));
      else
        return nullptr;

      if (CurTok == ")")
        break;

      getNextToken();
    }
  }

  // Eat the ')'.
  getNextToken();

  return std::make_unique<CallExprAST>(IdName, std::move(Args));
}

static std::unique_ptr<ExprAST> ParsePrimary() {

  if(CurTok == "tok_identifier")
    return ParseIdentifierExpr();
  else if(CurTok == "tok_number")
    return ParseNumberExpr();
  else if(CurTok == "(")
    return ParseParenExpr();
  
}

static std::unique_ptr<ExprAST> ParseBinOpRHS(int ExprPrec,
                                              std::unique_ptr<ExprAST> LHS) {
  // If this is a binop, find its precedence.
  while (true) {
    int TokPrec = GetTokPrecedence();

    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
    if (TokPrec < ExprPrec)
      return LHS;

    // Okay, we know this is a binop.
    
    int BinOp = BinOP[CurTok];
    getNextToken(); // eat binop

    // Parse the primary expression after the binary operator.
    auto RHS = ParsePrimary();
    if (!RHS)
      return nullptr;

    // If BinOp binds less tightly with RHS than the operator after RHS, let
    // the pending operator take RHS as its LHS.
    int NextPrec = GetTokPrecedence();
    if (TokPrec < NextPrec) {
      RHS = ParseBinOpRHS(TokPrec + 1, std::move(RHS));
      if (!RHS)
        return nullptr;
    }

    // Merge LHS/RHS.
    LHS =
        std::make_unique<BinaryExprAST>(BinOp, std::move(LHS), std::move(RHS));
  }
}

static std::unique_ptr<ExprAST> ParseExpression() {
  auto LHS = ParsePrimary();
  if (!LHS)
    return nullptr;

  return ParseBinOpRHS(0, std::move(LHS));
}

static std::unique_ptr<PrototypeAST> ParsePrototype() {
  if (CurTok != "tok_identifier")
    return LogErrorP("Expected function name in prototype");

  std::string FnName = identifier;
  getNextToken();

  if (CurTok != "(")
    return LogErrorP("Expected '(' in prototype");

  std::vector<std::string> ArgNames;
  while (getNextToken() == "tok_identifier")
    ArgNames.push_back(identifier);
  if (CurTok != ")")
    return LogErrorP("Expected ')' in prototype");

  // success.
  getNextToken(); // eat ')'.

  return std::make_unique<PrototypeAST>(FnName, std::move(ArgNames));
}

static std::unique_ptr<FunctionAST> ParseDefinition() {
  getNextToken(); // eat def.
  auto Proto = ParsePrototype();
  if (!Proto)
    return nullptr;

  if (auto E = ParseExpression())
    return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
  return nullptr;
}

static std::unique_ptr<FunctionAST> ParseTopLevelExpr() {
  if (auto E = ParseExpression()) {
    // Make an anonymous proto.
    auto Proto = std::make_unique<PrototypeAST>("__anon_expr",
                                                std::vector<std::string>());
    return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
  }
  return nullptr;
}

static std::unique_ptr<PrototypeAST> ParseExtern() {
  getNextToken(); // eat extern.
  return ParsePrototype();
}

static void HandleDefinition() {
  if (ParseDefinition()) {
    fprintf(stderr, "Parsed a function definition.\n");
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleExtern() {
  if (ParseExtern()) {
    fprintf(stderr, "Parsed an extern\n");
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleTopLevelExpression() {
  // Evaluate a top-level expression into an anonymous function.
  if (ParseTopLevelExpr()) {
    fprintf(stderr, "Parsed a top-level expr\n");
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void MainLoop() {
  while (true) {
    fprintf(stderr, "ready> ");
    if(CurTok == "tok_eof")
      return;
    else if(CurTok == "tok_next"){
      getNextToken();
      break;
    }
    else if(CurTok == "tok_def"){
      HandleDefinition();
      break;
    }
    else if(CurTok == "tok_extern"){
      HandleExtern();
      break;
    }else{
      HandleTopLevelExpression();
      break;
    }
  }
}

int main(){
    BinopPrecedence["大于"] = 10;
    BinopPrecedence["加"] = 20;
    BinopPrecedence["减"] = 20;
    BinopPrecedence["乘"] = 40; // highest.

  fprintf(stderr, "ready> ");
  getNextToken();

  // Run the main "interpreter loop" now.
  MainLoop();

  return 0;
    // if(isalpha(LastChar)){
    //     identifier = LastChar;
    //     while(isalnum((LastChar = getchar()))){
    //         identifier += LastChar;
    //     }
    //     printf("%s\n", identifier.c_str());
    // }
    
}


