#include <Setting.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
namespace rr {
const std::type_info& Setting::typeInfo() const {
    return std::visit([](auto&& x) -> decltype(auto) { return std::move(typeid(x)); }, value_);
}
}
__attribute__((noinline,noipa)) bool read_bool(const rr::Setting& s) { return s.getAs<bool>(); }
__attribute__((noinline,noipa)) double read_double(const rr::Setting& s) { return s.getAs<double>(); }
volatile double sink;
int main() {
 rr::Setting bs[] = {rr::Setting(false), rr::Setting(true)};
 rr::Setting ds[] = {rr::Setting(1.e-6), rr::Setting(1.)};
 // Validate exact values and special floating-point bit patterns on the affected path.
 for (bool b : {false,true}) if (read_bool(rr::Setting(b)) != b) return 2;
 for (double d : {0.,-0.,1.,-1.,1.e-300,INFINITY*1.,-INFINITY*1.,NAN*1.}) {
   double actual=read_double(rr::Setting(d));
   if (std::memcmp(&actual,&d,sizeof(d))) return 3;
 }
 constexpr int n=300000;
 std::cout<<std::setprecision(10)<<"{\"bool_ns\":[";
 for (int k=0;k<5;++k) {
  double sum=0;auto start=std::chrono::steady_clock::now();
  for(int j=0;j<n;++j)sum+=read_bool(bs[j&1]);
  auto stop=std::chrono::steady_clock::now();sink=sum;
  if(k)std::cout<<",";std::cout<<std::chrono::duration<double,std::nano>(stop-start).count()/n;
 }
 std::cout<<"],\"double_ns\":[";
 for (int k=0;k<5;++k) {
  double sum=0;auto start=std::chrono::steady_clock::now();
  for(int j=0;j<n;++j)sum+=read_double(ds[j&1]);
  auto stop=std::chrono::steady_clock::now();sink=sum;
  if(k)std::cout<<",";std::cout<<std::chrono::duration<double,std::nano>(stop-start).count()/n;
 }
 std::cout<<"],\"iterations_per_round\":"<<n<<",\"validation\":\"passed\"}\n";
}
