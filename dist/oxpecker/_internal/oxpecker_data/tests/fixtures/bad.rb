# bad Ruby: bare rescue, eval, puts, missing bang
def dangerous
  eval("puts 'hi'")
  user = User.find(1)
  user.save
rescue
  puts "something failed"
end
